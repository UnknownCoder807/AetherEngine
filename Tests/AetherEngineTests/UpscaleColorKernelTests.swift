import Foundation
import Testing
import CoreVideo
import Metal
@testable import AetherEngine

@Suite("Upscale colour kernels", .enabled(if: MTLCreateSystemDefaultDevice() != nil))
struct UpscaleColorKernelTests {

    @Test func roundTripIsWithinOneCodeValue() throws {
        let h = try KernelHarness()
        let (w, ht) = (64, 64)
        let src = try h.makePixelBuffer(width: w, height: ht, videoRange: true)
        // Smooth gradient: luma ramps across the picture, chroma stays near neutral and varies slowly
        // and the ranges stay inside the RGB cube so clamping does not hide a conversion error.
        h.fill(src) { x, y in
            let yv = UInt8(60 + (x * 130) / (w - 1))
            let cb = UInt8(112 + (y * 32) / (ht - 1))
            let cr = UInt8(144 - (x * 32) / (w - 1))
            return (yv, cb, cr)
        }
        let rgba = try h.makeRGBA(width: w, height: ht)
        let dst = try h.makePixelBuffer(width: w, height: ht, videoRange: true)
        try h.nv12ToRGBA(src, rgba: rgba, videoRange: true)
        try h.rgbaToNV12(rgba, dst: dst, videoRange: true)

        var maxY = 0, maxC = 0
        for y in 0..<ht {
            for x in 0..<w {
                maxY = max(maxY, abs(Int(h.luma(src, x, y)) - Int(h.luma(dst, x, y))))
            }
        }
        for y in 0..<(ht / 2) {
            for x in 0..<(w / 2) {
                let a = h.chroma(src, x, y), b = h.chroma(dst, x, y)
                maxC = max(maxC, abs(Int(a.0) - Int(b.0)), abs(Int(a.1) - Int(b.1)))
            }
        }
        #expect(maxY <= 1)
        #expect(maxC <= 1)
    }

    @Test func videoRangeBlackAndWhiteMapToRGBExtremes() throws {
        let h = try KernelHarness()
        let src = try h.makePixelBuffer(width: 16, height: 16, videoRange: true)
        h.fill(src) { x, _ in (x < 8 ? 16 : 235, 128, 128) }
        let rgba = try h.makeRGBA(width: 16, height: 16)
        try h.nv12ToRGBA(src, rgba: rgba, videoRange: true)
        let black = h.rgb(rgba, 2, 8), white = h.rgb(rgba, 12, 8)
        #expect(black.allSatisfy { $0 <= 1 })
        #expect(white.allSatisfy { $0 >= 254 })
    }

    @Test func fullRangeIsHonoured() throws {
        let h = try KernelHarness()
        let src = try h.makePixelBuffer(width: 16, height: 16, videoRange: false)
        h.fill(src) { x, _ in (x < 8 ? 0 : 255, 128, 128) }
        let rgba = try h.makeRGBA(width: 16, height: 16)
        try h.nv12ToRGBA(src, rgba: rgba, videoRange: false)
        let black = h.rgb(rgba, 2, 8), white = h.rgb(rgba, 12, 8)
        #expect(black.allSatisfy { $0 == 0 })
        #expect(white.allSatisfy { $0 == 255 })
    }
}

/// Builds the two pipelines and runs them over 420v/420f buffers; Task 3 owns the production dispatch.
private struct KernelHarness {
    struct Failure: Error {}
    let device: MTLDevice
    let queue: MTLCommandQueue
    let toRGBA: MTLComputePipelineState
    let toNV12: MTLComputePipelineState
    let cache: CVMetalTextureCache

    init() throws {
        guard let dev = MTLCreateSystemDefaultDevice(), let q = dev.makeCommandQueue() else { throw Failure() }
        let lib = try dev.makeLibrary(source: UpscaleShaderSource.source, options: nil)
        guard let f1 = lib.makeFunction(name: "nv12ToRGBA"), let f2 = lib.makeFunction(name: "rgbaToNV12") else { throw Failure() }
        device = dev
        queue = q
        toRGBA = try dev.makeComputePipelineState(function: f1)
        toNV12 = try dev.makeComputePipelineState(function: f2)
        var c: CVMetalTextureCache?
        guard CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, dev, nil, &c) == kCVReturnSuccess, let c else { throw Failure() }
        cache = c
    }

    func makePixelBuffer(width: Int, height: Int, videoRange: Bool) throws -> CVPixelBuffer {
        var pb: CVPixelBuffer?
        let fmt = videoRange ? kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange : kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
        let attrs: [CFString: Any] = [kCVPixelBufferMetalCompatibilityKey: true,
                                      kCVPixelBufferIOSurfacePropertiesKey: [String: Any]()]
        guard CVPixelBufferCreate(kCFAllocatorDefault, width, height, fmt, attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let pb else { throw Failure() }
        return pb
    }

    func makeRGBA(width: Int, height: Int) throws -> MTLTexture {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: width, height: height, mipmapped: false)
        d.usage = [.shaderRead, .shaderWrite]
        d.storageMode = .shared
        guard let t = device.makeTexture(descriptor: d) else { throw Failure() }
        return t
    }

    func fill(_ pb: CVPixelBuffer, _ f: (Int, Int) -> (UInt8, UInt8, UInt8)) {
        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }
        let w = CVPixelBufferGetWidthOfPlane(pb, 0), ht = CVPixelBufferGetHeightOfPlane(pb, 0)
        let ys = CVPixelBufferGetBytesPerRowOfPlane(pb, 0), cs = CVPixelBufferGetBytesPerRowOfPlane(pb, 1)
        let yp = CVPixelBufferGetBaseAddressOfPlane(pb, 0)!.assumingMemoryBound(to: UInt8.self)
        let cp = CVPixelBufferGetBaseAddressOfPlane(pb, 1)!.assumingMemoryBound(to: UInt8.self)
        for y in 0..<ht { for x in 0..<w { yp[y * ys + x] = f(x, y).0 } }
        for y in 0..<(ht / 2) {
            for x in 0..<(w / 2) {
                let v = f(x * 2, y * 2)
                cp[y * cs + x * 2] = v.1
                cp[y * cs + x * 2 + 1] = v.2
            }
        }
    }

    func luma(_ pb: CVPixelBuffer, _ x: Int, _ y: Int) -> UInt8 {
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        let p = CVPixelBufferGetBaseAddressOfPlane(pb, 0)!.assumingMemoryBound(to: UInt8.self)
        return p[y * CVPixelBufferGetBytesPerRowOfPlane(pb, 0) + x]
    }

    func chroma(_ pb: CVPixelBuffer, _ x: Int, _ y: Int) -> (UInt8, UInt8) {
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        let p = CVPixelBufferGetBaseAddressOfPlane(pb, 1)!.assumingMemoryBound(to: UInt8.self)
        let o = y * CVPixelBufferGetBytesPerRowOfPlane(pb, 1) + x * 2
        return (p[o], p[o + 1])
    }

    func rgb(_ t: MTLTexture, _ x: Int, _ y: Int) -> [UInt8] {
        var px = [UInt8](repeating: 0, count: 4)
        t.getBytes(&px, bytesPerRow: 4, from: MTLRegionMake2D(x, y, 1, 1), mipmapLevel: 0)
        return Array(px[0..<3])
    }

    private func planeTexture(_ pb: CVPixelBuffer, plane: Int, format: MTLPixelFormat) throws -> (MTLTexture, CVMetalTexture) {
        var tex: CVMetalTexture?
        let w = CVPixelBufferGetWidthOfPlane(pb, plane), h = CVPixelBufferGetHeightOfPlane(pb, plane)
        guard CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, cache, pb, nil, format, w, h, plane, &tex) == kCVReturnSuccess,
              let tex, let mt = CVMetalTextureGetTexture(tex) else { throw Failure() }
        return (mt, tex)
    }

    private func run(_ pipeline: MTLComputePipelineState, grid: (Int, Int), videoRange: Bool, textures: [MTLTexture]) throws {
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { throw Failure() }
        enc.setComputePipelineState(pipeline)
        for (i, t) in textures.enumerated() { enc.setTexture(t, index: i) }
        var range: UInt32 = videoRange ? 1 : 0
        enc.setBytes(&range, length: 4, index: 0)
        let tg = MTLSize(width: 8, height: 8, depth: 1)
        enc.dispatchThreads(MTLSize(width: grid.0, height: grid.1, depth: 1), threadsPerThreadgroup: tg)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if cb.status != .completed { throw Failure() }
    }

    func nv12ToRGBA(_ pb: CVPixelBuffer, rgba: MTLTexture, videoRange: Bool) throws {
        let y = try planeTexture(pb, plane: 0, format: .r8Unorm)
        let c = try planeTexture(pb, plane: 1, format: .rg8Unorm)
        try run(toRGBA, grid: (rgba.width, rgba.height), videoRange: videoRange, textures: [y.0, c.0, rgba])
        withExtendedLifetime((y.1, c.1)) {}
    }

    func rgbaToNV12(_ rgba: MTLTexture, dst: CVPixelBuffer, videoRange: Bool) throws {
        let y = try planeTexture(dst, plane: 0, format: .r8Unorm)
        let c = try planeTexture(dst, plane: 1, format: .rg8Unorm)
        try run(toNV12, grid: (rgba.width / 2, rgba.height / 2), videoRange: videoRange, textures: [rgba, y.0, c.0])
        withExtendedLifetime((y.1, c.1)) {}
    }
}
