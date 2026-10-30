#if !targetEnvironment(simulator)
import Foundation
import Testing
import CoreGraphics
import CoreVideo
import Metal
import VideoToolbox
@testable import AetherEngine

private enum UpscalerGate {
    static var gpuAvailable: Bool {
        guard #available(macOS 26, iOS 26, tvOS 26, visionOS 26, *) else { return false }
        return MTLCreateSystemDefaultDevice() != nil
    }

    static var mlAvailable: Bool {
        guard #available(macOS 26, iOS 26, tvOS 26, visionOS 26, *) else { return false }
        return VTLowLatencySuperResolutionScalerConfiguration.isSupported
    }
}

/// One frame in, one frame out: sizes, colour tags, flat-field fidelity, mid-stream size changes and
/// the disable paths. GPU and OS 26 gated; the ML tests also need VT low-latency super resolution.
@Suite("Frame upscaler", .serialized, .enabled(if: UpscalerGate.gpuAvailable))
struct FrameUpscalerTests {

    private static let uhd = CGSize(width: 3840, height: 2160)

    private func plan(_ w: Int, _ h: Int, sar: (Int, Int) = (1, 1), display: CGSize = uhd, ml: [Float]) throws -> UpscalePlan {
        try #require(UpscaleRouteDecision.makePlan(
            source: SourceGeometry(width: w, height: h, sarNum: sar.0, sarDen: sar.1),
            display: display, mlScaleFactors: { _, _ in ml }))
    }

    @available(macOS 26, iOS 26, tvOS 26, visionOS 26, *)
    private func upscaler(_ plan: UpscalePlan, frameInterval: Double = 1.0 / 24) throws -> FrameUpscaler {
        try #require(FrameUpscaler(plan: plan, frameInterval: frameInterval))
    }

    @Test(.enabled(if: UpscalerGate.mlAvailable))
    @available(macOS 26, iOS 26, tvOS 26, visionOS 26, *)
    func sd2xPlusMetalFXDeliversTargetSize() throws {
        let p = try plan(720, 480, sar: (8, 9), ml: [2])
        #expect(p.mlScale == 2 && p.targetWidth == 2880 && p.targetHeight == 2160 && p.needsMetalFX)
        let up = try upscaler(p)
        let input = try TestFrames.nv12(720, 480, par: (8, 9))
        for _ in 0..<10 {
            let out = try #require(up.upscale(input))
            #expect(CVPixelBufferGetWidth(out) == 2880 && CVPixelBufferGetHeight(out) == 2160)
            #expect(CVPixelBufferGetPixelFormatType(out) == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange)
        }
        let s = up.stats
        #expect(s.framesUpscaled == 10 && s.failures == 0)
        print(String(format: "[FrameUpscalerTests] 720x480 ML 2x + MetalFX -> 2880x2160: mean %.2f ms, max %.2f ms", s.meanMs, s.maxMs))
    }

    @Test(.enabled(if: UpscalerGate.mlAvailable))
    @available(macOS 26, iOS 26, tvOS 26, visionOS 26, *)
    func outputCarriesSquarePixelsAndInputColour() throws {
        let up = try upscaler(try plan(720, 480, sar: (8, 9), ml: [2]))
        let input = try TestFrames.nv12(720, 480, par: (8, 9))
        let out = try #require(up.upscale(input))
        let par = try #require(CVBufferCopyAttachment(out, kCVImageBufferPixelAspectRatioKey, nil) as? [String: Any])
        #expect(par[kCVImageBufferPixelAspectRatioHorizontalSpacingKey as String] as? Int == 1)
        #expect(par[kCVImageBufferPixelAspectRatioVerticalSpacingKey as String] as? Int == 1)
        for key in [kCVImageBufferColorPrimariesKey, kCVImageBufferTransferFunctionKey, kCVImageBufferYCbCrMatrixKey] {
            let a = CVBufferCopyAttachment(input, key, nil) as? String
            let b = CVBufferCopyAttachment(out, key, nil) as? String
            #expect(a != nil && a == b, "\(key)")
        }
    }

    @Test(.enabled(if: UpscalerGate.mlAvailable))
    @available(macOS 26, iOS 26, tvOS 26, visionOS 26, *)
    func mlOnlyPlanSkipsMetalFX() throws {
        let p = try plan(1280, 720, display: CGSize(width: 1920, height: 1080), ml: [1.5])
        #expect(p.mlScale == 1.5 && !p.needsMetalFX)
        let up = try upscaler(p)
        let out = try #require(up.upscale(try TestFrames.nv12(1280, 720)))
        #expect(CVPixelBufferGetWidth(out) == 1920 && CVPixelBufferGetHeight(out) == 1080)
        print(String(format: "[FrameUpscalerTests] 1280x720 ML 1.5x -> 1920x1080: %.2f ms", up.stats.meanMs))
    }

    @Test
    @available(macOS 26, iOS 26, tvOS 26, visionOS 26, *)
    func metalFXOnlyPlanSkipsML() throws {
        let p = try plan(720, 480, sar: (8, 9), ml: [])
        #expect(p.mlScale == nil && p.needsMetalFX)
        let up = try upscaler(p)
        let input = try TestFrames.nv12(720, 480, par: (8, 9))
        for _ in 0..<5 {
            let out = try #require(up.upscale(input))
            #expect(CVPixelBufferGetWidth(out) == 2880 && CVPixelBufferGetHeight(out) == 2160)
        }
        print(String(format: "[FrameUpscalerTests] 720x480 MetalFX -> 2880x2160: mean %.2f ms", up.stats.meanMs))
    }

    /// MetalFX alone keeps every sample. VT's super resolution rings in the outermost ~12 source pixels
    /// even on a flat field (measured: Y 126 comes back as 49...154 at the very edge, the deviation scaling
    /// with the level, as if the model pads with zeros), and its interior carries a +-1 texture that
    /// MetalFX's sharpening lifts to +-3, so the ML pipeline is held to the interior: no shift in the
    /// mean, nothing past 3 code values.
    @Test
    @available(macOS 26, iOS 26, tvOS 26, visionOS 26, *)
    func flatGreyStaysFlatGrey() throws {
        let grey = try TestFrames.nv12(720, 480, y: 126, c: 128)
        let mfx = try #require(try upscaler(try plan(720, 480, sar: (8, 9), ml: [])).upscale(grey))
        let all = TestFrames.planeRange(mfx)
        #expect(all.y.lowerBound >= 124 && all.y.upperBound <= 128, "MetalFX Y \(all.y)")
        #expect(all.c.lowerBound >= 126 && all.c.upperBound <= 130, "MetalFX C \(all.c)")

        guard UpscalerGate.mlAvailable else { return }
        let both = try #require(try upscaler(try plan(720, 480, sar: (8, 9), ml: [2])).upscale(grey))
        let inner = TestFrames.planeRange(both, margin: 32)
        #expect(inner.y.lowerBound >= 123 && inner.y.upperBound <= 129, "ML+MetalFX Y \(inner.y)")
        #expect(inner.c.lowerBound >= 125 && inner.c.upperBound <= 131, "ML+MetalFX C \(inner.c)")
        #expect(abs(inner.meanY - 126) < 0.5, "ML+MetalFX mean Y \(inner.meanY)")
        #expect(abs(inner.meanC - 128) < 0.5, "ML+MetalFX mean C \(inner.meanC)")
    }

    @Test
    @available(macOS 26, iOS 26, tvOS 26, visionOS 26, *)
    func reconfiguresOnInputSizeChange() throws {
        let up = try upscaler(try plan(720, 480, sar: (8, 9), ml: FrameUpscaler.mlScaleFactors(720, 480)))
        let first = try #require(up.upscale(try TestFrames.nv12(720, 480, par: (8, 9))))
        #expect(CVPixelBufferGetWidth(first) == 2880 && CVPixelBufferGetHeight(first) == 2160)
        let second = try #require(up.upscale(try TestFrames.nv12(704, 576)))
        #expect(CVPixelBufferGetWidth(second) == 2640 && CVPixelBufferGetHeight(second) == 2160)
        #expect(!up.isDisabled)
    }

    @Test
    @available(macOS 26, iOS 26, tvOS 26, visionOS 26, *)
    func unsupportedFormatPassesThrough() throws {
        let up = try upscaler(try plan(720, 480, ml: []))
        let tenBit = try TestFrames.buffer(720, 480, format: kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange)
        #expect(up.upscale(tenBit) == nil)
        #expect(!up.isDisabled)
        #expect(up.upscale(try TestFrames.nv12(720, 480)) != nil)
    }

    @Test
    @available(macOS 26, iOS 26, tvOS 26, visionOS 26, *)
    func budgetGuardDisablesAfterSustainedOverrun() throws {
        let up = try upscaler(try plan(720, 480, ml: []), frameInterval: 0.0001)
        let events = EventLog()
        up.onEvent = { events.add($0) }
        let input = try TestFrames.nv12(720, 480)
        for i in 0..<47 {
            #expect(up.upscale(input) != nil, "frame \(i)")
            #expect(!up.isDisabled)
        }
        _ = up.upscale(input)
        #expect(up.isDisabled)
        #expect(up.upscale(input) == nil)
        let all = events.all
        #expect(all.count == 2)
        if case .firstFrame = all.first {} else { Issue.record("first event \(String(describing: all.first))") }
        if case let .disabled(_, stats) = all.last { #expect(stats.framesUpscaled == 48) } else {
            Issue.record("last event \(String(describing: all.last))")
        }
    }
}

@available(macOS 26, iOS 26, tvOS 26, visionOS 26, *)
private final class EventLog: @unchecked Sendable {
    private let lock = NSLock()
    private var events: [FrameUpscaler.Event] = []
    func add(_ e: FrameUpscaler.Event) { lock.lock(); events.append(e); lock.unlock() }
    var all: [FrameUpscaler.Event] { lock.lock(); defer { lock.unlock() }; return events }
}

private enum TestFrames {
    struct Failure: Error {}

    static func buffer(_ w: Int, _ h: Int, format: OSType) throws -> CVPixelBuffer {
        var pb: CVPixelBuffer?
        let attrs: [CFString: Any] = [kCVPixelBufferMetalCompatibilityKey: true,
                                      kCVPixelBufferIOSurfacePropertiesKey: [String: Any]()]
        guard CVPixelBufferCreate(kCFAllocatorDefault, w, h, format, attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let pb else { throw Failure() }
        return pb
    }

    /// A 420v BT.709 frame, a soft gradient unless `y`/`c` pin it flat.
    static func nv12(_ w: Int, _ h: Int, par: (Int, Int)? = nil, y fixedY: UInt8? = nil, c fixedC: UInt8? = nil) throws -> CVPixelBuffer {
        let pb = try buffer(w, h, format: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange)
        CVBufferSetAttachment(pb, kCVImageBufferColorPrimariesKey, kCVImageBufferColorPrimaries_ITU_R_709_2, .shouldPropagate)
        CVBufferSetAttachment(pb, kCVImageBufferTransferFunctionKey, kCVImageBufferTransferFunction_ITU_R_709_2, .shouldPropagate)
        CVBufferSetAttachment(pb, kCVImageBufferYCbCrMatrixKey, kCVImageBufferYCbCrMatrix_ITU_R_709_2, .shouldPropagate)
        if let par {
            let dict: [String: Any] = [kCVImageBufferPixelAspectRatioHorizontalSpacingKey as String: par.0,
                                       kCVImageBufferPixelAspectRatioVerticalSpacingKey as String: par.1]
            CVBufferSetAttachment(pb, kCVImageBufferPixelAspectRatioKey, dict as CFDictionary, .shouldPropagate)
        }
        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }
        let ys = CVPixelBufferGetBytesPerRowOfPlane(pb, 0), cs = CVPixelBufferGetBytesPerRowOfPlane(pb, 1)
        let yp = CVPixelBufferGetBaseAddressOfPlane(pb, 0)!.assumingMemoryBound(to: UInt8.self)
        let cp = CVPixelBufferGetBaseAddressOfPlane(pb, 1)!.assumingMemoryBound(to: UInt8.self)
        for row in 0..<h {
            for x in 0..<w { yp[row * ys + x] = fixedY ?? UInt8(40 + (x * 160) / max(w - 1, 1)) }
        }
        for row in 0..<(h / 2) {
            for x in 0..<(w / 2) {
                cp[row * cs + x * 2] = fixedC ?? UInt8(112 + (row * 32) / max(h / 2 - 1, 1))
                cp[row * cs + x * 2 + 1] = fixedC ?? UInt8(144 - (x * 32) / max(w / 2 - 1, 1))
            }
        }
        return pb
    }

    /// Min/max of each plane, skipping `margin` luma pixels at every edge.
    static func planeRange(_ pb: CVPixelBuffer, margin: Int = 0)
        -> (y: ClosedRange<Int>, c: ClosedRange<Int>, meanY: Double, meanC: Double) {
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        func scan(_ plane: Int, _ bytes: Int) -> (ClosedRange<Int>, Double) {
            let p = CVPixelBufferGetBaseAddressOfPlane(pb, plane)!.assumingMemoryBound(to: UInt8.self)
            let stride = CVPixelBufferGetBytesPerRowOfPlane(pb, plane)
            let m = plane == 0 ? margin : margin / 2
            let w = CVPixelBufferGetWidthOfPlane(pb, plane), h = CVPixelBufferGetHeightOfPlane(pb, plane)
            var lo = 255, hi = 0, sum = 0, count = 0
            for row in m..<(h - m) {
                for x in (m * bytes)..<((w - m) * bytes) {
                    let v = Int(p[row * stride + x])
                    lo = min(lo, v); hi = max(hi, v); sum += v; count += 1
                }
            }
            return (lo...hi, Double(sum) / Double(max(count, 1)))
        }
        let y = scan(0, 1), c = scan(1, 2)
        return (y.0, c.0, y.1, c.1)
    }
}
#endif
