import Testing
@testable import AetherEngine

struct SnappierSeekClockTests {
    @Test func earlyDemuxLandingWaitsForAudioBeyondTheSeekTarget() throws {
        let host = SoftwareClockHarness()
        let output = try #require(host.audioOutput)
        host.completeSeek()
        #expect(output.rate == 0)
        #expect(host.publishedBuffering == [true])
        #expect(host.pauseArmOpen == false)

        for pts in [Double.nan, 74, 100, 101.99] {
            host.lastEnqueuedAudioPtsSec = pts
            host.applyAudioClockAction()
            #expect(output.rate == 0)
            #expect(output.currentTimeSeconds == 100)
            #expect(host.rebuffering)
        }
        host.lastEnqueuedAudioPtsSec = 102
        host.applyAudioClockAction()
        #expect(output.rate == 1)
        #expect(host.rebuffering == false)
        #expect(host.publishedBuffering == [true, false])
    }

    @Test func slowSourceStillUsesUpstreamStarvationRecovery() throws {
        let host = SoftwareClockHarness()
        let output = try #require(host.audioOutput)
        host.lastEnqueuedAudioPtsSec = 100.1
        host.applyAudioClockAction()
        #expect(output.rate == 1)
        #expect(host.pauseArmOpen == false)
        host.lastEnqueuedAudioPtsSec = 98.9
        host.applyAudioClockAction()
        #expect(host.pauseArmOpen)
        #expect(output.rate == 0)
        host.lastEnqueuedAudioPtsSec = 102
        host.applyAudioClockAction()
        #expect(output.rate == 1)
        #expect(host.publishedBuffering == [true, false])
    }

    @Test func pausedSeekDoesNotStartARebufferResume() throws {
        let host = SoftwareClockHarness()
        let output = try #require(host.audioOutput)
        host.playing = false
        output.pause()
        host.completeSeek()
        host.lastEnqueuedAudioPtsSec = 103
        host.applyAudioClockAction()
        #expect(output.rate == 0)
        #expect(host.rebuffering == false)
        #expect(host.publishedBuffering.isEmpty)
    }

    @Test func anotherSeekReplacesThePreviousBufferingHold() throws {
        let host = SoftwareClockHarness()
        let output = try #require(host.audioOutput)
        host.completeSeek()
        output.currentTimeSeconds = 200
        host.completeSeek(generation: 2)
        host.lastEnqueuedAudioPtsSec = 102
        host.applyAudioClockAction()
        #expect(output.rate == 0)
        host.lastEnqueuedAudioPtsSec = 202
        host.applyAudioClockAction()
        #expect(output.rate == 1)
        #expect(host.publishedBuffering == [true, false, true, false])
    }
}
