import Foundation
import CoreMedia
import CoreVideo
import AetherLibavformat
import AetherLibavcodec

/// Runs the upscale stage (#846) on every frame a decoder delivers. Everything but the frame handler
/// forwards to `inner`; a nil from the stage delivers the original frame.
final class UpscalingVideoDecoder: VideoDecodingPipeline, @unchecked Sendable {

    private let inner: any VideoDecodingPipeline
    private let upscale: @Sendable (CVPixelBuffer) -> CVPixelBuffer?

    init(inner: any VideoDecodingPipeline, upscale: @escaping @Sendable (CVPixelBuffer) -> CVPixelBuffer?) {
        self.inner = inner
        self.upscale = upscale
    }

    var onFrame: DecodedFrameHandler? {
        get { inner.onFrame }
        set { inner.onFrame = newValue }
    }
    var onFirstHDR10PlusDetected: (@Sendable () -> Void)? {
        get { inner.onFirstHDR10PlusDetected }
        set { inner.onFirstHDR10PlusDetected = newValue }
    }
    var onA53Captions: (@Sendable ([CCDataParser.CCTriplet], Double) -> Void)? {
        get { inner.onA53Captions }
        set { inner.onA53Captions = newValue }
    }
    var onDecodedFormat: (@Sendable (DecodedVideoFormat) -> Void)? {
        get { inner.onDecodedFormat }
        set { inner.onDecodedFormat = newValue }
    }
    var skipUntilPTS: CMTime? {
        get { inner.skipUntilPTS }
        set { inner.skipUntilPTS = newValue }
    }
    var framesDecoded: UInt64 { inner.framesDecoded }
    var feedEpoch: UInt64 { inner.feedEpoch }

    func open(stream: UnsafeMutablePointer<AVStream>, onFrame: @escaping DecodedFrameHandler) throws {
        let upscale = self.upscale
        try inner.open(stream: stream) { buffer, pts, hdr10Plus in
            onFrame(upscale(buffer) ?? buffer, pts, hdr10Plus)
        }
    }

    func decode(packet: UnsafeMutablePointer<AVPacket>, epoch: UInt64?) { inner.decode(packet: packet, epoch: epoch) }
    func flush() { inner.flush() }
    func drainForEndOfStream() { inner.drainForEndOfStream() }
    func close() { inner.close() }
}
