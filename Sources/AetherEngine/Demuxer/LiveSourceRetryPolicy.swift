import Foundation

/// Pacing for a live HTTP source after a retryable refusal. Shared by both Apple platforms.
enum LiveSourceRetryPolicy {
    static let maxRetries = 5
    static let budgetSeconds: TimeInterval = 60

    static func delay(streak: Int, retryAfter: TimeInterval) -> TimeInterval {
        let exponential = pow(2, Double(max(1, min(streak, 4)) - 1))
        let requested = retryAfter.isFinite ? max(0, retryAfter) : 0
        return max(exponential, requested)
    }

    /// A server delay that will not fit is a terminal result, never a shorter wait.
    static func shouldRetry(streak: Int, retryAfter: TimeInterval,
                            elapsed: TimeInterval, budget: TimeInterval,
                            scale: Double = 1) -> Bool {
        guard (1...maxRetries).contains(streak), elapsed < budget else { return false }
        return delay(streak: streak, retryAfter: retryAfter) * scale < budget - elapsed
    }

    static func parseRetryAfter(_ value: String?, now: Date = .now) -> TimeInterval {
        guard let value else { return 0 }
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return 0 }
        if trimmed.utf8.allSatisfy({ (48...57).contains($0) }),
           let seconds = TimeInterval(trimmed), seconds.isFinite {
            return seconds
        }
        // Response parsers accept HTTP-date, including the two obsolete HTTP date forms.
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.isLenient = false
        for format in ["EEE, dd MMM yyyy HH:mm:ss zzz", "EEEE, dd-MMM-yy HH:mm:ss zzz", "EEE MMM d HH:mm:ss yyyy"] {
            formatter.dateFormat = format
            if let date = formatter.date(from: trimmed) {
                return max(0, date.timeIntervalSince(now))
            }
        }
        return 0
    }
}
