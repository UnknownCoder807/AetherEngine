import Testing
import CoreGraphics
import AetherLibavutil
@testable import AetherEngine

/// #846: the load side of the upscale stage. The option is off by default and correctable, the SDR test
/// refuses only PQ and HLG, and the published status follows the stage's own events. The inventory of
/// `LoadOptions` fields and the documentation of every public symbol are held by
/// `Issue460ReloadWithOptionsTests.fieldInventoryIsComplete` and `PublicAPIDocumentationTests`.
@Suite("Upscale load integration")
struct UpscaleLoadIntegrationTests {

    private let plan = UpscalePlan(
        sourceWidth: 720, sourceHeight: 480, mlScale: 2,
        mlOutputWidth: 1440, mlOutputHeight: 960, targetWidth: 2880, targetHeight: 2160,
        displayWidth: 3840, displayHeight: 2160)

    @Test("the option defaults off and is a correctable field, not an identity one")
    func optionDefaultsOffAndIsCorrectable() {
        #expect(LoadOptions().upscalesLowResolutionVideo == false)
        #expect(SessionOptionCorrection.knownFields.contains("upscalesLowResolutionVideo"))
        #expect(!SessionOptionCorrection.loadIdentityFields.contains("upscalesLowResolutionVideo"))
        var proposed = LoadOptions()
        proposed.upscalesLowResolutionVideo = true
        #expect(SessionOptionCorrection.refusedFields(from: LoadOptions(), to: proposed).isEmpty)
    }

    @Test("only PQ and HLG are refused; an unknown transfer counts as SDR",
          arguments: [
            (AVCOL_TRC_BT709, true), (AVCOL_TRC_UNSPECIFIED, true), (AVCOL_TRC_RESERVED0, true),
            (AVCOL_TRC_SMPTE170M, true), (AVCOL_TRC_GAMMA22, true),
            (AVCOL_TRC_SMPTE2084, false), (AVCOL_TRC_ARIB_STD_B67, false),
          ])
    func sdrTransfer(trc: AVColorTransferCharacteristic, sdr: Bool) {
        #expect(UpscaleRouteDecision.isSDR(colorTransfer: trc) == sdr)
    }

    @Test("the first frame publishes the plan with no disabled reason")
    func firstFramePublishesThePlan() {
        let status = UpscaleStatus.afterFirstFrame(plan, current: nil)
        #expect(status.sourceSize == CGSize(width: 720, height: 480))
        #expect(status.mlScale == 2)
        #expect(status.targetSize == CGSize(width: 2880, height: 2160))
        #expect(status.disabledReason == nil)
    }

    @Test("a disable republishes the status with its reason, and a late first frame does not clear it")
    func disableStaysPublished() {
        let disabled = UpscaleStatus.afterDisable(reason: "3 consecutive failures", current: nil, plan: plan)
        #expect(disabled.disabledReason == "3 consecutive failures")
        #expect(disabled.targetSize == CGSize(width: 2880, height: 2160))
        #expect(UpscaleStatus.afterFirstFrame(plan, current: disabled) == disabled)
    }

    @Test("a disable keeps the plan the first frame published over the one the load decided")
    func disableKeepsThePublishedPlan() {
        var rebuilt = plan
        rebuilt.sourceWidth = 640
        let published = UpscaleStatus.afterFirstFrame(rebuilt, current: nil)
        let disabled = UpscaleStatus.afterDisable(reason: "over budget", current: published, plan: plan)
        #expect(disabled.sourceSize == CGSize(width: 640, height: 480))
        #expect(disabled.disabledReason == "over budget")
    }
}
