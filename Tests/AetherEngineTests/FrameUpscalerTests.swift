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
@Suite("Frame upscaler", .serialized, .timeLimit(.minutes(1)), .enabled(if: UpscalerGate.gpuAvailable))
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

    /// MetalFX alone keeps every sample. VT's ML pass carries a +-1 texture (MetalFX sharpening lifts it a
    /// little) and its edge ringing is clamped away, so the ML paths get +-4 with no border exemption.
    @Test
    @available(macOS 26, iOS 26, tvOS 26, visionOS 26, *)
    func flatGreyStaysFlatGrey() throws {
        let grey = try TestFrames.nv12(720, 480, y: 126, c: 128)
        let mfx = try #require(try upscaler(try plan(720, 480, sar: (8, 9), ml: [])).upscale(grey))
        let all = TestFrames.planeRange(mfx)
        #expect(all.y.lowerBound >= 124 && all.y.upperBound <= 128, "MetalFX Y \(all.y)")
        #expect(all.c.lowerBound >= 126 && all.c.upperBound <= 130, "MetalFX C \(all.c)")

        guard UpscalerGate.mlAvailable else { return }
        let mlOnly = try upscaler(try plan(720, 480, display: CGSize(width: 1440, height: 960), ml: [2]))
        let both = try upscaler(try plan(720, 480, sar: (8, 9), ml: [2]))
        for (name, up) in [("ML", mlOnly), ("ML+MetalFX", both)] {
            let out = try #require(up.upscale(grey))
            let r = TestFrames.planeRange(out)
            #expect(r.y.lowerBound >= 122 && r.y.upperBound <= 130, "\(name) Y \(r.y)")
            #expect(abs(r.meanY - 126) < 0.5, "\(name) mean Y \(r.meanY)")
            #expect(abs(r.meanC - 128) < 0.5, "\(name) mean C \(r.meanC)")
        }
    }

    /// The edge clamp is pinned on a vertical gradient: the outer 12 rows repeat row 12 (top) and row
    /// h-13 (bottom), the outer 12 columns repeat column 12 / w-13.
    @Test(.enabled(if: UpscalerGate.mlAvailable))
    @available(macOS 26, iOS 26, tvOS 26, visionOS 26, *)
    func mlOutputBorderIsClamped() throws {
        let up = try upscaler(try plan(1280, 720, display: CGSize(width: 1920, height: 1080), ml: [1.5]))
        let out = try #require(up.upscale(try TestFrames.nv12(1280, 720, vertical: true)))
        CVPixelBufferLockBaseAddress(out, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(out, .readOnly) }
        let p = CVPixelBufferGetBaseAddressOfPlane(out, 0)!.assumingMemoryBound(to: UInt8.self)
        let s = CVPixelBufferGetBytesPerRowOfPlane(out, 0)
        let w = CVPixelBufferGetWidth(out), h = CVPixelBufferGetHeight(out)
        func y(_ x: Int, _ r: Int) -> UInt8 { p[r * s + x] }
        #expect(y(w / 2, 12) != y(w / 2, h - 13), "the input must not be flat")
        var mismatches = 0
        for r in 0..<12 {
            for x in 0..<w where y(x, r) != y(x, 12) || y(x, h - 1 - r) != y(x, h - 13) { mismatches += 1 }
        }
        for r in 0..<h {
            for x in 0..<12 where y(x, r) != y(12, r) || y(w - 1 - x, r) != y(w - 13, r) { mismatches += 1 }
        }
        #expect(mismatches == 0)
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

    /// A cropped stream: the decoder buffer has padding rows below a clean aperture. The aperture is the
    /// picture, the padding (white here) must not reach the output, and the output carries no aperture or
    /// field tags of the input's geometry.
    @Test
    @available(macOS 26, iOS 26, tvOS 26, visionOS 26, *)
    func upscalesTheCleanApertureOnly() throws {
        let up = try upscaler(try plan(720, 480, sar: (8, 9), ml: []))
        let input = try TestFrames.nv12(720, 496, par: (8, 9), y: 126, c: 128)
        TestFrames.paint(input, rows: 480..<496, y: 235)
        let aperture: [String: Any] = [kCVImageBufferCleanApertureWidthKey as String: 720,
                                       kCVImageBufferCleanApertureHeightKey as String: 480,
                                       kCVImageBufferCleanApertureHorizontalOffsetKey as String: 0,
                                       kCVImageBufferCleanApertureVerticalOffsetKey as String: -8]
        CVBufferSetAttachment(input, kCVImageBufferCleanApertureKey, aperture as CFDictionary, .shouldPropagate)
        CVBufferSetAttachment(input, kCVImageBufferFieldCountKey, 1 as CFNumber, .shouldPropagate)

        let out = try #require(up.upscale(input))
        #expect(CVPixelBufferGetWidth(out) == 2880 && CVPixelBufferGetHeight(out) == 2160)
        #expect(up.plan.sourceWidth == 720 && up.plan.sourceHeight == 480)
        #expect(CVBufferCopyAttachment(out, kCVImageBufferCleanApertureKey, nil) == nil)
        #expect(CVBufferCopyAttachment(out, kCVImageBufferFieldCountKey, nil) == nil)
        #expect(CVBufferCopyAttachment(out, kCVImageBufferYCbCrMatrixKey, nil) != nil)
        let range = TestFrames.planeRange(out)
        #expect(range.y.lowerBound >= 124 && range.y.upperBound <= 128, "Y \(range.y)")
    }

    /// A VT pass that never calls back (session invalidated by backgrounding, media-services reset) must
    /// not hold the frame thread: bounded wait, original frame delivered, and the abandoned destination
    /// never handed out later.
    @Test(.enabled(if: UpscalerGate.mlAvailable))
    @available(macOS 26, iOS 26, tvOS 26, visionOS 26, *)
    func hungMLPassTimesOutAndNeverReusesItsBuffer() throws {
        let up = try upscaler(try plan(1280, 720, display: CGSize(width: 1920, height: 1080), ml: [1.5]),
                              frameInterval: 1.0 / 60)
        let hung = BufferLog()
        up.mlSubmitOverride = { params, _ in if let b = params.destinationFrame?.buffer { hung.add(b) } }
        let input = try TestFrames.nv12(1280, 720)
        for _ in 0..<2 {
            let t0 = Date()
            #expect(up.upscale(input) == nil)
            #expect(Date().timeIntervalSince(t0) < 0.5)
        }
        #expect(up.stats.failures == 2 && !up.isDisabled)
        up.mlSubmitOverride = nil
        for _ in 0..<6 {
            let out = try #require(up.upscale(input))
            #expect(!hung.contains(out))
        }
        #expect(hung.count == 2)
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

private final class BufferLog: @unchecked Sendable {
    private let lock = NSLock()
    private var buffers: [CVPixelBuffer] = []
    func add(_ b: CVPixelBuffer) { lock.lock(); buffers.append(b); lock.unlock() }
    func contains(_ b: CVPixelBuffer) -> Bool { lock.lock(); defer { lock.unlock() }; return buffers.contains { $0 === b } }
    var count: Int { lock.lock(); defer { lock.unlock() }; return buffers.count }
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
    static func nv12(_ w: Int, _ h: Int, par: (Int, Int)? = nil, y fixedY: UInt8? = nil, c fixedC: UInt8? = nil,
                     vertical: Bool = false) throws -> CVPixelBuffer {
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
            for x in 0..<w { yp[row * ys + x] = fixedY ?? UInt8(40 + ((vertical ? row : x) * 160) / max((vertical ? h : w) - 1, 1)) }
        }
        for row in 0..<(h / 2) {
            for x in 0..<(w / 2) {
                cp[row * cs + x * 2] = fixedC ?? UInt8(112 + (row * 32) / max(h / 2 - 1, 1))
                cp[row * cs + x * 2 + 1] = fixedC ?? UInt8(144 - (x * 32) / max(w / 2 - 1, 1))
            }
        }
        return pb
    }

    static func paint(_ pb: CVPixelBuffer, rows: Range<Int>, y value: UInt8) {
        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }
        let stride = CVPixelBufferGetBytesPerRowOfPlane(pb, 0)
        let p = CVPixelBufferGetBaseAddressOfPlane(pb, 0)!.assumingMemoryBound(to: UInt8.self)
        for row in rows { memset(p + row * stride, Int32(value), CVPixelBufferGetWidth(pb)) }
    }

    /// Min/max and mean of each plane.
    static func planeRange(_ pb: CVPixelBuffer)
        -> (y: ClosedRange<Int>, c: ClosedRange<Int>, meanY: Double, meanC: Double) {
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        func scan(_ plane: Int, _ bytes: Int) -> (ClosedRange<Int>, Double) {
            let p = CVPixelBufferGetBaseAddressOfPlane(pb, plane)!.assumingMemoryBound(to: UInt8.self)
            let stride = CVPixelBufferGetBytesPerRowOfPlane(pb, plane)
            let w = CVPixelBufferGetWidthOfPlane(pb, plane), h = CVPixelBufferGetHeightOfPlane(pb, plane)
            var lo = 255, hi = 0, sum = 0, count = 0
            for row in 0..<h {
                for x in 0..<(w * bytes) {
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
