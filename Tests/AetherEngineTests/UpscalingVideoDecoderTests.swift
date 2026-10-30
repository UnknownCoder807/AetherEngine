import Testing
import Foundation
import CoreMedia
import CoreVideo
import Synchronization
import AetherLibavformat
import AetherLibavcodec
@testable import AetherEngine

private final class FakeInnerDecoder: VideoDecodingPipeline, @unchecked Sendable {
    var onFrame: DecodedFrameHandler?
    var onFirstHDR10PlusDetected: (@Sendable () -> Void)?
    var onA53Captions: (@Sendable ([CCDataParser.CCTriplet], Double) -> Void)?
    var onDecodedFormat: (@Sendable (DecodedVideoFormat) -> Void)?
    var skipUntilPTS: CMTime?
    var framesDecoded: UInt64 = 7
    var feedEpoch: UInt64 = 3
    var flushCount = 0
    var closeCount = 0
    var drainCount = 0
    var handler: DecodedFrameHandler?

    func open(stream: UnsafeMutablePointer<AVStream>, onFrame: @escaping DecodedFrameHandler) throws {
        handler = onFrame
    }
    func decode(packet: UnsafeMutablePointer<AVPacket>, epoch: UInt64?) {}
    func flush() { flushCount += 1; feedEpoch += 1 }
    func drainForEndOfStream() { drainCount += 1 }
    func close() { closeCount += 1 }
}

private func makeBuffer(width: Int) -> CVPixelBuffer {
    var buffer: CVPixelBuffer?
    CVPixelBufferCreate(nil, width, 16, kCVPixelFormatType_32BGRA, nil, &buffer)
    return buffer!
}

@Suite("Upscaling video decoder")
struct UpscalingVideoDecoderTests {

    private func open(_ decoder: UpscalingVideoDecoder, collect: @escaping DecodedFrameHandler) throws {
        let stream = UnsafeMutablePointer<AVStream>.allocate(capacity: 1)
        defer { stream.deallocate() }
        try decoder.open(stream: stream, onFrame: collect)
    }

    @Test func framesPassThroughTheStageInOrder() throws {
        let inner = FakeInnerDecoder()
        let a = makeBuffer(width: 16), b = makeBuffer(width: 32), c = makeBuffer(width: 48)
        let decoder = UpscalingVideoDecoder(inner: inner) { makeBuffer(width: CVPixelBufferGetWidth($0) * 10) }
        let seen = Mutex<[(Int, Double, Data?)]>([])
        try open(decoder) { buf, pts, data in
            seen.withLock { $0.append((CVPixelBufferGetWidth(buf), pts.seconds, data)) }
        }
        let tag = Data([1, 2, 3])
        inner.handler?(a, CMTime(seconds: 1, preferredTimescale: 600), nil)
        inner.handler?(b, CMTime(seconds: 2, preferredTimescale: 600), tag)
        inner.handler?(c, CMTime(seconds: 3, preferredTimescale: 600), nil)
        let got = seen.withLock { $0 }
        #expect(got.map(\.0) == [160, 320, 480])
        #expect(got.map(\.1) == [1, 2, 3])
        #expect(got[1].2 == tag)
    }

    @Test func nilFromTheStageDeliversTheOriginal() throws {
        let inner = FakeInnerDecoder()
        let decoder = UpscalingVideoDecoder(inner: inner) { _ in nil }
        let original = makeBuffer(width: 16)
        let originalID = ObjectIdentifier(original)
        let seen = Mutex<[Bool]>([])
        try open(decoder) { buf, _, _ in seen.withLock { $0.append(ObjectIdentifier(buf) == originalID) } }
        inner.handler?(original, .zero, nil)
        #expect(seen.withLock { $0 } == [true])
    }

    @Test func flushForwardsAndDropsNothingStale() {
        let inner = FakeInnerDecoder()
        let decoder = UpscalingVideoDecoder(inner: inner) { $0 }
        #expect(decoder.feedEpoch == 3)
        #expect(decoder.framesDecoded == 7)
        decoder.flush()
        decoder.drainForEndOfStream()
        decoder.close()
        #expect(inner.flushCount == 1)
        #expect(inner.drainCount == 1)
        #expect(inner.closeCount == 1)
        #expect(decoder.feedEpoch == 4)
        decoder.skipUntilPTS = CMTime(seconds: 5, preferredTimescale: 600)
        #expect(inner.skipUntilPTS?.seconds == 5)
    }

    #if !targetEnvironment(simulator)
    @MainActor
    @Test func hostWrapsOnlyWithAPlan() throws {
        guard #available(tvOS 26, iOS 26, macOS 26, visionOS 26, *) else { return }
        let host = SoftwarePlaybackHost()
        let plain = host.selectVideoDecoder(codecID: AV_CODEC_ID_AV1, canHardwareDecode: false, dvProfile: nil)
        #expect(!(plain is UpscalingVideoDecoder))

        host.upscalePlan = UpscalePlan(
            sourceWidth: 1280, sourceHeight: 720, mlScale: nil, mlOutputWidth: 1280, mlOutputHeight: 720,
            targetWidth: 1920, targetHeight: 1080, displayWidth: 1920, displayHeight: 1080)
        let wrapped = host.selectVideoDecoder(codecID: AV_CODEC_ID_AV1, canHardwareDecode: false, dvProfile: nil, frameRate: 24)
        #expect(wrapped is UpscalingVideoDecoder)
        #expect(host.frameUpscaler != nil)
    }
    #endif
}
