import Foundation
import CoreMedia
import CoreVideo

/// Runs the upscale stage (#846) on every frame handed to a decoder's frame handler; a nil from the
/// stage delivers the original frame.
enum UpscalingFrameHandler {
    static func wrap(
        _ handler: @escaping DecodedFrameHandler,
        upscale: @escaping @Sendable (CVPixelBuffer) -> CVPixelBuffer?
    ) -> DecodedFrameHandler {
        { buffer, pts, hdr10Plus in
            handler(upscale(buffer) ?? buffer, pts, hdr10Plus)
        }
    }
}
