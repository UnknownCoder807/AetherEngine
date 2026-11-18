import Foundation
import Testing
@testable import AetherEngine

struct LiveSourceRetryPolicyTests {
    @Test func exponentialPacingIsCapped() {
        #expect((1...8).map { LiveSourceRetryPolicy.delay(streak: $0, retryAfter: 0) }
                == [1, 2, 4, 8, 8, 8, 8, 8])
        #expect(LiveSourceRetryPolicy.delay(streak: Int.max, retryAfter: 0) == 8)
        #expect(LiveSourceRetryPolicy.delay(streak: Int.min, retryAfter: 0) == 1)
    }

    @Test(arguments: [30.0, 120.0, 3600.0])
    func serverDelayIsNeverShortened(seconds: Double) {
        #expect(LiveSourceRetryPolicy.delay(streak: 5, retryAfter: seconds) == seconds)
    }

    @Test func boundedCountAndElapsedBudget() {
        #expect(LiveSourceRetryPolicy.shouldRetry(streak: 5, retryAfter: 0, elapsed: 20, budget: 60))
        #expect(!LiveSourceRetryPolicy.shouldRetry(streak: 6, retryAfter: 0, elapsed: 20, budget: 60))
        #expect(!LiveSourceRetryPolicy.shouldRetry(streak: 1, retryAfter: 0, elapsed: 60, budget: 60))
        #expect(!LiveSourceRetryPolicy.shouldRetry(streak: 3, retryAfter: 30, elapsed: 35, budget: 60))
        #expect(!LiveSourceRetryPolicy.shouldRetry(streak: 1, retryAfter: 120, elapsed: 0, budget: 60))
    }

    @Test func retryAfterParsesSecondsAndHTTPDate() {
        let now = Date(timeIntervalSince1970: 784111777) // Sun, 06 Nov 1994 08:49:37 GMT
        #expect(LiveSourceRetryPolicy.parseRetryAfter("120", now: now) == 120)
        #expect(LiveSourceRetryPolicy.parseRetryAfter("Sun, 06 Nov 1994 08:51:37 GMT", now: now) == 120)
        #expect(LiveSourceRetryPolicy.parseRetryAfter("Sunday, 06-Nov-94 08:51:37 GMT", now: now) == 120)
        #expect(LiveSourceRetryPolicy.parseRetryAfter("Sun Nov 6 08:51:37 1994", now: now) == 120)
        #expect(LiveSourceRetryPolicy.parseRetryAfter("Sun, 06 Nov 1994 08:48:37 GMT", now: now) == 0)
    }

    @Test(arguments: ["", "-1", "nan", "inf", "2.5", "tomorrow"])
    func invalidHeadersFallBackToPacing(header: String) {
        #expect(LiveSourceRetryPolicy.parseRetryAfter(header) == 0)
    }

    @Test func missingHeaderAndNonfiniteWaitAreSafe() {
        #expect(LiveSourceRetryPolicy.parseRetryAfter(nil) == 0)
        #expect(LiveSourceRetryPolicy.delay(streak: 2, retryAfter: .nan) == 2)
        #expect(LiveSourceRetryPolicy.delay(streak: 2, retryAfter: .infinity) == 2)
    }
}
