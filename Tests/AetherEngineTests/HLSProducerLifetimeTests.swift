#if os(tvOS)
import Synchronization
import Testing
@testable import AetherEngine

struct HLSProducerLifetimeTests {
    @Test func cleanupWaitsForTheRetiredPumpAndItsReplacement() throws {
        let lifetime = HLSProducerLifetime()
        let cleanups = Mutex(0)
        try #require(lifetime.begin())
        try #require(lifetime.begin())
        lifetime.close { cleanups.withLock { $0 += 1 } }
        #expect(!lifetime.begin())
        lifetime.end()
        #expect(cleanups.withLock { $0 } == 0)
        lifetime.end()
        #expect(cleanups.withLock { $0 } == 1)
    }

    @Test(arguments: [false, true])
    func cleanupCanRunImmediatelyWhenNoPumpIsRunning(started: Bool) throws {
        let lifetime = HLSProducerLifetime()
        if started {
            try #require(lifetime.begin())
            lifetime.end()
        }
        let cleaned = Mutex(false)
        lifetime.close { cleaned.withLock { $0 = true } }
        #expect(cleaned.withLock { $0 })
        #expect(!lifetime.begin())
    }
}
#endif
