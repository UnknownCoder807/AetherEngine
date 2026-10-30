// VTLowLatencySuperResolutionScaler is not in the simulator SDKs.
#if !targetEnvironment(simulator)
import Foundation
import CoreGraphics
import CoreMedia
import CoreVideo
import Metal
import MetalFX
import VideoToolbox

/// The per-frame upscale stage (#846): optional VT low-latency super resolution, then MetalFX Spatial to
/// the plan's target size. Synchronous; `upscale` returns nil whenever the caller should deliver the
/// original frame. Failures and a sustained overrun disable it for good, each logged once.
@available(tvOS 26, iOS 26, macOS 26, visionOS 26, *)
final class FrameUpscaler: @unchecked Sendable {

    struct Stats: Equatable, Sendable {
        var framesUpscaled: Int
        var failures: Int
        var meanMs: Double
        var maxMs: Double
    }

    enum Event: Sendable {
        case firstFrame(UpscalePlan)
        case disabled(reason: String, stats: Stats)
    }

    static let budgetWindow = 48
    static let budgetFraction = 0.75
    static let maxConsecutiveFailures = 3

    /// Scale factors VT offers for a frame size, empty when the ML scaler is unsupported.
    static func mlScaleFactors(_ width: Int, _ height: Int) -> [Float] {
        guard VTLowLatencySuperResolutionScalerConfiguration.isSupported else { return [] }
        return VTLowLatencySuperResolutionScalerConfiguration
            .__supportedScaleFactors(forFrameWidth: width, frameHeight: height).map(\.floatValue)
    }

    var onEvent: (@Sendable (Event) -> Void)? {
        get { stateLock.lock(); defer { stateLock.unlock() }; return _onEvent }
        set { stateLock.lock(); _onEvent = newValue; stateLock.unlock() }
    }

    var isDisabled: Bool { stateLock.lock(); defer { stateLock.unlock() }; return _isDisabled }

    var stats: Stats { stateLock.lock(); defer { stateLock.unlock() }; return _stats }

    var plan: UpscalePlan { processLock.lock(); defer { processLock.unlock() }; return stage.plan }

    private let frameInterval: Double
    private let gpu: GPU
    private var stage: Stage
    private var lastPassThroughNote: String?
    private var recentMs: [Double] = []
    private var totalMs = 0.0
    private var consecutiveFailures = 0
    private var firstFrameSent = false
    private var frameIndex: Int64 = 0
    private let processLock = NSLock()

    private let stateLock = NSLock()
    private var _onEvent: (@Sendable (Event) -> Void)?
    private var _isDisabled = false
    private var _stats = Stats(framesUpscaled: 0, failures: 0, meanMs: 0, maxMs: 0)

    init?(plan: UpscalePlan, frameInterval: Double) {
        guard let gpu = GPU() else {
            EngineLog.emit("[Upscale] stage unavailable: no Metal device or kernels", category: .swPlayback)
            return nil
        }
        do {
            stage = try Stage(plan: plan, gpu: gpu)
        } catch {
            EngineLog.emit("[Upscale] stage unavailable: \(error)", category: .swPlayback)
            return nil
        }
        self.gpu = gpu
        self.frameInterval = frameInterval
    }

    func upscale(_ input: CVPixelBuffer) -> CVPixelBuffer? {
        processLock.lock()
        defer { processLock.unlock() }
        guard !isDisabled else { return nil }

        let format = CVPixelBufferGetPixelFormatType(input)
        guard format == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
                || format == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange else {
            notePassThrough("pixel format \(fourCC(format)) is not 8-bit 4:2:0")
            return nil
        }
        let width = CVPixelBufferGetWidth(input), height = CVPixelBufferGetHeight(input)
        if width != stage.plan.sourceWidth || height != stage.plan.sourceHeight {
            guard rebuild(for: input, width: width, height: height) else { return nil }
        }
        lastPassThroughNote = nil

        let start = DispatchTime.now().uptimeNanoseconds
        frameIndex += 1
        do {
            let output = try stage.process(input, frameIndex: frameIndex)
            let ms = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
            recordSuccess(ms: ms)
            return output
        } catch {
            recordFailure(error)
            return nil
        }
    }

    // MARK: - Rebuild and pass-through

    private func rebuild(for input: CVPixelBuffer, width: Int, height: Int) -> Bool {
        guard width < 1920, height < 1080 else {
            notePassThrough("\(width)x\(height) is not below 1080p")
            return false
        }
        let sar = Self.pixelAspectRatio(of: input)
        let display = CGSize(width: stage.plan.displayWidth, height: stage.plan.displayHeight)
        guard let plan = UpscaleRouteDecision.makePlan(
                source: SourceGeometry(width: width, height: height, sarNum: sar.0, sarDen: sar.1),
                display: display, mlScaleFactors: Self.mlScaleFactors),
              plan.targetWidth > width || plan.targetHeight > height else {
            notePassThrough("\(width)x\(height) is not smaller than the display")
            return false
        }
        do {
            stage = try Stage(plan: plan, gpu: gpu)
        } catch {
            disable(reason: "rebuild for \(width)x\(height) failed: \(error)")
            return false
        }
        EngineLog.emit("[Upscale] input size changed: \(width)x\(height) -> \(plan.targetWidth)x\(plan.targetHeight)"
                       + (plan.mlScale.map { " (ML \(String(format: "%g", $0))x)" } ?? ""), category: .swPlayback)
        return true
    }

    private func notePassThrough(_ note: String) {
        guard note != lastPassThroughNote else { return }
        lastPassThroughNote = note
        EngineLog.emit("[Upscale] passing frames through: \(note)", category: .swPlayback)
    }

    private static func pixelAspectRatio(of buffer: CVPixelBuffer) -> (Int, Int) {
        guard let par = CVBufferCopyAttachment(buffer, kCVImageBufferPixelAspectRatioKey, nil) as? [String: Any],
              let h = (par[kCVImageBufferPixelAspectRatioHorizontalSpacingKey as String] as? NSNumber)?.intValue,
              let v = (par[kCVImageBufferPixelAspectRatioVerticalSpacingKey as String] as? NSNumber)?.intValue
        else { return (1, 1) }
        return (h, v)
    }

    // MARK: - Accounting

    private func recordSuccess(ms: Double) {
        consecutiveFailures = 0
        totalMs += ms
        recentMs.append(ms)
        if recentMs.count > Self.budgetWindow { recentMs.removeFirst() }
        stateLock.lock()
        _stats.framesUpscaled += 1
        _stats.meanMs = totalMs / Double(_stats.framesUpscaled)
        _stats.maxMs = max(_stats.maxMs, ms)
        let handler = _onEvent
        stateLock.unlock()

        if !firstFrameSent {
            firstFrameSent = true
            handler?(.firstFrame(stage.plan))
        }
        let budgetMs = Self.budgetFraction * frameInterval * 1000
        if recentMs.count == Self.budgetWindow {
            let windowMean = recentMs.reduce(0, +) / Double(recentMs.count)
            if windowMean > budgetMs {
                disable(reason: String(format: "over budget, %.2f ms mean over %d frames against %.2f ms",
                                       windowMean, Self.budgetWindow, budgetMs))
            }
        }
    }

    private func recordFailure(_ error: Error) {
        consecutiveFailures += 1
        stateLock.lock()
        _stats.failures += 1
        stateLock.unlock()
        if consecutiveFailures >= Self.maxConsecutiveFailures {
            disable(reason: "\(consecutiveFailures) consecutive failures, last: \(error)")
        }
    }

    private func disable(reason: String) {
        stateLock.lock()
        guard !_isDisabled else { stateLock.unlock(); return }
        _isDisabled = true
        let stats = _stats
        let handler = _onEvent
        stateLock.unlock()
        EngineLog.emit(String(format: "[Upscale] disabled: %@ (frames %d, failures %d, mean %.2f ms, max %.2f ms)",
                              reason, stats.framesUpscaled, stats.failures, stats.meanMs, stats.maxMs),
                       category: .swPlayback)
        handler?(.disabled(reason: reason, stats: stats))
        stage.release()
    }
}

// MARK: - GPU state shared across rebuilds

@available(tvOS 26, iOS 26, macOS 26, visionOS 26, *)
private struct UpscaleError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

@available(tvOS 26, iOS 26, macOS 26, visionOS 26, *)
private final class GPU {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let toRGBA: MTLComputePipelineState
    let toNV12: MTLComputePipelineState
    let textureCache: CVMetalTextureCache

    init?() {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue(),
              let library = try? device.makeLibrary(source: UpscaleShaderSource.source, options: nil),
              let f1 = library.makeFunction(name: "nv12ToRGBA"), let f2 = library.makeFunction(name: "rgbaToNV12"),
              let toRGBA = try? device.makeComputePipelineState(function: f1),
              let toNV12 = try? device.makeComputePipelineState(function: f2)
        else { return nil }
        var cache: CVMetalTextureCache?
        guard CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &cache) == kCVReturnSuccess,
              let cache else { return nil }
        self.device = device
        self.queue = queue
        self.toRGBA = toRGBA
        self.toNV12 = toNV12
        self.textureCache = cache
    }

    func planeTexture(_ buffer: CVPixelBuffer, plane: Int, format: MTLPixelFormat) throws -> (MTLTexture, CVMetalTexture) {
        var ref: CVMetalTexture?
        let w = CVPixelBufferGetWidthOfPlane(buffer, plane), h = CVPixelBufferGetHeightOfPlane(buffer, plane)
        guard CVMetalTextureCacheCreateTextureFromImage(
                kCFAllocatorDefault, textureCache, buffer, nil, format, w, h, plane, &ref) == kCVReturnSuccess,
              let ref, let texture = CVMetalTextureGetTexture(ref)
        else { throw UpscaleError("no Metal texture for plane \(plane)") }
        return (texture, ref)
    }

    func dispatch(_ encoder: MTLComputeCommandEncoder, _ pipeline: MTLComputePipelineState,
                  textures: [MTLTexture], videoRange: Bool, grid: (Int, Int)) {
        encoder.setComputePipelineState(pipeline)
        for (i, t) in textures.enumerated() { encoder.setTexture(t, index: i) }
        var range: UInt32 = videoRange ? 1 : 0
        encoder.setBytes(&range, length: MemoryLayout<UInt32>.size, index: 0)
        let w = pipeline.threadExecutionWidth
        let h = max(1, pipeline.maxTotalThreadsPerThreadgroup / w)
        encoder.dispatchThreads(MTLSize(width: grid.0, height: grid.1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: w, height: h, depth: 1))
    }
}

// MARK: - One plan's processing chain

@available(tvOS 26, iOS 26, macOS 26, visionOS 26, *)
private final class Stage {
    let plan: UpscalePlan
    private let gpu: GPU
    private let ml: MLScaler?
    private let scaler: MTLFXSpatialScaler?
    private let rgba: MTLTexture?
    private let scaled: MTLTexture?
    private var outputPools: [OSType: CVPixelBufferPool] = [:]

    init(plan: UpscalePlan, gpu: GPU) throws {
        self.plan = plan
        self.gpu = gpu
        ml = try plan.mlScale.map { try MLScaler(width: plan.sourceWidth, height: plan.sourceHeight, scale: $0) }

        let inW = ml?.outputWidth ?? plan.sourceWidth
        let inH = ml?.outputHeight ?? plan.sourceHeight
        guard inW != plan.targetWidth || inH != plan.targetHeight else {
            guard ml != nil else { throw UpscaleError("plan scales nothing") }
            scaler = nil; rgba = nil; scaled = nil
            return
        }
        guard MTLFXSpatialScalerDescriptor.supportsDevice(gpu.device) else {
            throw UpscaleError("MetalFX spatial scaling unsupported")
        }
        let d = MTLFXSpatialScalerDescriptor()
        d.colorTextureFormat = .rgba8Unorm
        d.outputTextureFormat = .rgba8Unorm
        d.inputWidth = inW
        d.inputHeight = inH
        d.outputWidth = plan.targetWidth
        d.outputHeight = plan.targetHeight
        d.colorProcessingMode = .perceptual
        guard let scaler = d.makeSpatialScaler(device: gpu.device) else {
            throw UpscaleError("MetalFX scaler \(inW)x\(inH) -> \(plan.targetWidth)x\(plan.targetHeight) not built")
        }
        let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: inW, height: inH, mipmapped: false)
        td.usage = scaler.colorTextureUsage.union(.shaderWrite)
        td.storageMode = .private
        let od = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: plan.targetWidth, height: plan.targetHeight, mipmapped: false)
        od.usage = scaler.outputTextureUsage.union(.shaderRead)
        od.storageMode = .private
        guard let rgba = gpu.device.makeTexture(descriptor: td), let scaled = gpu.device.makeTexture(descriptor: od) else {
            throw UpscaleError("intermediate textures not allocated")
        }
        scaler.colorTexture = rgba
        scaler.outputTexture = scaled
        scaler.inputContentWidth = inW
        scaler.inputContentHeight = inH
        self.scaler = scaler
        self.rgba = rgba
        self.scaled = scaled
    }

    deinit { release() }

    func release() { ml?.end() }

    func process(_ input: CVPixelBuffer, frameIndex: Int64) throws -> CVPixelBuffer {
        let format = CVPixelBufferGetPixelFormatType(input)
        let videoRange = format == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        let mlOut = try ml?.process(input, frameIndex: frameIndex)
        let output: CVPixelBuffer

        if let scaler, let rgba, let scaled {
            output = try makeOutput(format: format, width: plan.targetWidth, height: plan.targetHeight)
            let source = mlOut ?? input
            // The ML pass carries sample values through unchanged, so its output keeps the input's range.
            let sy = try gpu.planeTexture(source, plane: 0, format: .r8Unorm)
            let sc = try gpu.planeTexture(source, plane: 1, format: .rg8Unorm)
            let oy = try gpu.planeTexture(output, plane: 0, format: .r8Unorm)
            let oc = try gpu.planeTexture(output, plane: 1, format: .rg8Unorm)
            guard let cb = gpu.queue.makeCommandBuffer() else { throw UpscaleError("no command buffer") }
            guard let enc = cb.makeComputeCommandEncoder() else { throw UpscaleError("no encoder") }
            gpu.dispatch(enc, gpu.toRGBA, textures: [sy.0, sc.0, rgba], videoRange: videoRange,
                         grid: (rgba.width, rgba.height))
            enc.endEncoding()
            scaler.encode(commandBuffer: cb)
            guard let enc2 = cb.makeComputeCommandEncoder() else { throw UpscaleError("no encoder") }
            gpu.dispatch(enc2, gpu.toNV12, textures: [scaled, oy.0, oc.0], videoRange: videoRange,
                         grid: (scaled.width / 2, scaled.height / 2))
            enc2.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            withExtendedLifetime((sy.1, sc.1, oy.1, oc.1)) {}
            CVMetalTextureCacheFlush(gpu.textureCache, 0)
            guard cb.status == .completed else {
                throw UpscaleError("command buffer \(cb.status.rawValue): \(cb.error.map { "\($0)" } ?? "no error")")
            }
        } else if let mlOut {
            if CVPixelBufferGetPixelFormatType(mlOut) == format {
                output = mlOut
            } else {
                output = try makeOutput(format: format, width: CVPixelBufferGetWidth(mlOut), height: CVPixelBufferGetHeight(mlOut))
                try copyPlanes(from: mlOut, to: output)
            }
        } else {
            throw UpscaleError("plan has no stage")
        }

        CVBufferRemoveAllAttachments(output)
        if let attachments = CVBufferCopyAttachments(input, .shouldPropagate) {
            CVBufferSetAttachments(output, attachments, .shouldPropagate)
        }
        let square: [String: Int] = [kCVImageBufferPixelAspectRatioHorizontalSpacingKey as String: 1,
                                     kCVImageBufferPixelAspectRatioVerticalSpacingKey as String: 1]
        CVBufferSetAttachment(output, kCVImageBufferPixelAspectRatioKey, square as CFDictionary, .shouldPropagate)
        return output
    }

    private func makeOutput(format: OSType, width: Int, height: Int) throws -> CVPixelBuffer {
        if outputPools[format] == nil {
            outputPools[format] = try makePool(width: width, height: height, format: format)
        }
        return try pooledBuffer(outputPools[format]!)
    }
}

// MARK: - VT low-latency super resolution

@available(tvOS 26, iOS 26, macOS 26, visionOS 26, *)
private final class MLScaler {
    let outputWidth: Int
    let outputHeight: Int
    private let processor = VTFrameProcessor()
    private let sourcePool: CVPixelBufferPool
    private let destinationPool: CVPixelBufferPool
    private let sourceFormat: OSType
    private var copiesInput = false
    private var ended = false

    init(width: Int, height: Int, scale: Float) throws {
        let config = VTLowLatencySuperResolutionScalerConfiguration(frameWidth: width, frameHeight: height, scaleFactor: scale)
        let src = config.sourcePixelBufferAttributes
        let dst = config.destinationPixelBufferAttributes
        outputWidth = (dst[kCVPixelBufferWidthKey as String] as? NSNumber)?.intValue ?? Int((Float(width) * scale).rounded())
        outputHeight = (dst[kCVPixelBufferHeightKey as String] as? NSNumber)?.intValue ?? Int((Float(height) * scale).rounded())
        sourceFormat = (src[kCVPixelBufferPixelFormatTypeKey as String] as? NSNumber)?.uint32Value
            ?? kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        sourcePool = try makePool(attributes: src, width: width, height: height)
        destinationPool = try makePool(attributes: dst, width: outputWidth, height: outputHeight)
        try processor.startSession(configuration: config)
    }

    deinit { end() }

    func end() {
        guard !ended else { return }
        ended = true
        processor.endSession()
    }

    func process(_ input: CVPixelBuffer, frameIndex: Int64) throws -> CVPixelBuffer {
        if !copiesInput, CVPixelBufferGetPixelFormatType(input) == sourceFormat, CVPixelBufferGetIOSurface(input) != nil {
            do {
                return try run(input, frameIndex: frameIndex)
            } catch {
                copiesInput = true
                EngineLog.emit("[Upscale] ML scaler refused the decoder buffer (\(error)), copying input from now on",
                               category: .swPlayback)
            }
        }
        let copy = try pooledBuffer(sourcePool)
        try copyPlanes(from: input, to: copy)
        return try run(copy, frameIndex: frameIndex)
    }

    private func run(_ source: CVPixelBuffer, frameIndex: Int64) throws -> CVPixelBuffer {
        let destination = try pooledBuffer(destinationPool)
        let pts = CMTime(value: frameIndex, timescale: 600)
        guard let sf = VTFrameProcessorFrame(buffer: source, presentationTimeStamp: pts),
              let df = VTFrameProcessorFrame(buffer: destination, presentationTimeStamp: pts)
        else { throw UpscaleError("VTFrameProcessorFrame refused a buffer") }
        let params = VTLowLatencySuperResolutionScalerParameters(sourceFrame: sf, destinationFrame: df)
        let done = Completion()
        processor.process(parameters: params) { _, error in done.finish(error) }
        if let error = done.wait() { throw error }
        return destination
    }
}

@available(tvOS 26, iOS 26, macOS 26, visionOS 26, *)
private final class Completion: @unchecked Sendable {
    private let semaphore = DispatchSemaphore(value: 0)
    private var error: Error?
    func finish(_ error: Error?) { self.error = error; semaphore.signal() }
    func wait() -> Error? { semaphore.wait(); return error }
}

// MARK: - Buffers

@available(tvOS 26, iOS 26, macOS 26, visionOS 26, *)
private func makePool(width: Int, height: Int, format: OSType) throws -> CVPixelBufferPool {
    try makePool(attributes: [kCVPixelBufferPixelFormatTypeKey as String: format], width: width, height: height)
}

@available(tvOS 26, iOS 26, macOS 26, visionOS 26, *)
private func makePool(attributes: [String: Any], width: Int, height: Int) throws -> CVPixelBufferPool {
    var attrs = attributes
    attrs[kCVPixelBufferWidthKey as String] = width
    attrs[kCVPixelBufferHeightKey as String] = height
    attrs[kCVPixelBufferMetalCompatibilityKey as String] = true
    if attrs[kCVPixelBufferIOSurfacePropertiesKey as String] == nil {
        attrs[kCVPixelBufferIOSurfacePropertiesKey as String] = [String: Any]()
    }
    var pool: CVPixelBufferPool?
    let status = CVPixelBufferPoolCreate(
        kCFAllocatorDefault, [kCVPixelBufferPoolMinimumBufferCountKey as String: 3] as CFDictionary,
        attrs as CFDictionary, &pool)
    guard status == kCVReturnSuccess, let pool else { throw UpscaleError("pixel buffer pool \(width)x\(height): \(status)") }
    return pool
}

@available(tvOS 26, iOS 26, macOS 26, visionOS 26, *)
private func pooledBuffer(_ pool: CVPixelBufferPool) throws -> CVPixelBuffer {
    var buffer: CVPixelBuffer?
    let status = CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &buffer)
    guard status == kCVReturnSuccess, let buffer else { throw UpscaleError("pool buffer: \(status)") }
    return buffer
}

/// Plane-by-plane copy between two 8-bit bi-planar buffers of the same size; the values carry over as is.
@available(tvOS 26, iOS 26, macOS 26, visionOS 26, *)
private func copyPlanes(from src: CVPixelBuffer, to dst: CVPixelBuffer) throws {
    guard CVPixelBufferGetWidth(src) == CVPixelBufferGetWidth(dst),
          CVPixelBufferGetHeight(src) == CVPixelBufferGetHeight(dst),
          CVPixelBufferGetPlaneCount(src) == 2, CVPixelBufferGetPlaneCount(dst) == 2
    else { throw UpscaleError("plane copy between mismatched buffers") }
    CVPixelBufferLockBaseAddress(src, .readOnly)
    CVPixelBufferLockBaseAddress(dst, [])
    defer {
        CVPixelBufferUnlockBaseAddress(dst, [])
        CVPixelBufferUnlockBaseAddress(src, .readOnly)
    }
    for plane in 0..<2 {
        guard let s = CVPixelBufferGetBaseAddressOfPlane(src, plane),
              let d = CVPixelBufferGetBaseAddressOfPlane(dst, plane)
        else { throw UpscaleError("plane \(plane) has no base address") }
        let sStride = CVPixelBufferGetBytesPerRowOfPlane(src, plane)
        let dStride = CVPixelBufferGetBytesPerRowOfPlane(dst, plane)
        let rowBytes = min(sStride, dStride, CVPixelBufferGetWidthOfPlane(src, plane) * (plane == 0 ? 1 : 2))
        for row in 0..<CVPixelBufferGetHeightOfPlane(src, plane) {
            memcpy(d + row * dStride, s + row * sStride, rowBytes)
        }
    }
}

private func fourCC(_ code: OSType) -> String {
    let bytes = [24, 16, 8, 0].map { UInt8((code >> $0) & 0xFF) }
    return bytes.allSatisfy { $0 >= 32 && $0 < 127 } ? String(decoding: bytes, as: UTF8.self) : String(code)
}
#endif
