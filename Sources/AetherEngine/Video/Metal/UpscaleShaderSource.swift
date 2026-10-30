/// BT.709 NV12 <-> RGBA kernels around MetalFX Spatial, which takes and returns RGBA. SDR only, so no
/// transfer or primaries handling: the conversion is the plain matrix on the coded (gamma) values.
enum UpscaleShaderSource {
    static let source = """
    #include <metal_stdlib>
    using namespace metal;

    struct UpscaleColorParams { uint videoRange; };

    constant float3 kYCbCrToRGB_R = float3(1.0, 0.0, 1.5748);
    constant float3 kYCbCrToRGB_G = float3(1.0, -0.187324, -0.468124);
    constant float3 kYCbCrToRGB_B = float3(1.0, 1.8556, 0.0);

    inline float3 decodeYCbCr(float y, float2 c, bool videoRange) {
        float3 v;
        if (videoRange) {
            v = float3((y * 255.0 - 16.0) / 219.0, (c * 255.0 - 128.0) / 224.0);
        } else {
            v = float3(y, c - 128.0 / 255.0);
        }
        return float3(dot(kYCbCrToRGB_R, v), dot(kYCbCrToRGB_G, v), dot(kYCbCrToRGB_B, v));
    }

    inline float3 encodeYCbCr(float3 rgb, bool videoRange) {
        float y = dot(rgb, float3(0.2126, 0.7152, 0.0722));
        float cb = (rgb.b - y) / 1.8556;
        float cr = (rgb.r - y) / 1.5748;
        if (videoRange) {
            return float3((16.0 + 219.0 * y) / 255.0, (128.0 + 224.0 * cb) / 255.0, (128.0 + 224.0 * cr) / 255.0);
        }
        return float3(y, cb + 128.0 / 255.0, cr + 128.0 / 255.0);
    }

    kernel void nv12ToRGBA(constant UpscaleColorParams &p [[buffer(0)]],
                           texture2d<float, access::read> yTex [[texture(0)]],
                           texture2d<float, access::read> cTex [[texture(1)]],
                           texture2d<float, access::write> outTex [[texture(2)]],
                           uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= outTex.get_width() || gid.y >= outTex.get_height()) { return; }
        float y = yTex.read(gid).r;
        float2 c = cTex.read(gid / 2).rg;
        float3 rgb = saturate(decodeYCbCr(y, c, p.videoRange == 1));
        outTex.write(float4(rgb, 1.0), gid);
    }

    // One thread per 2x2 luma block: writes its four Y samples and one CbCr from the averaged RGB.
    kernel void rgbaToNV12(constant UpscaleColorParams &p [[buffer(0)]],
                           texture2d<float, access::read> inTex [[texture(0)]],
                           texture2d<float, access::write> yTex [[texture(1)]],
                           texture2d<float, access::write> cTex [[texture(2)]],
                           uint2 gid [[thread_position_in_grid]]) {
        uint2 base = gid * 2;
        if (base.x >= inTex.get_width() || base.y >= inTex.get_height()) { return; }
        bool videoRange = p.videoRange == 1;
        float3 sum = float3(0.0);
        for (uint dy = 0; dy < 2; dy++) {
            for (uint dx = 0; dx < 2; dx++) {
                uint2 pos = base + uint2(dx, dy);
                float3 rgb = inTex.read(pos).rgb;
                sum += rgb;
                yTex.write(float4(encodeYCbCr(rgb, videoRange).x, 0.0, 0.0, 1.0), pos);
            }
        }
        float3 e = encodeYCbCr(sum * 0.25, videoRange);
        cTex.write(float4(e.y, e.z, 0.0, 1.0), gid);
    }
    """
}
