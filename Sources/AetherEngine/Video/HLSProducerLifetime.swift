#if os(tvOS)
import Synchronization

/// Tracks the pump threads and seek restarts borrowing one HLS session's resources.
/// A retired producer can still be running after its replacement is installed.
final class HLSProducerLifetime: Sendable {
    private struct State: Sendable {
        var users = 0
        var closed = false
        var onDrain: (@Sendable () -> Void)?
    }

    private let state = Mutex(State())

    func begin() -> Bool {
        state.withLock { state in
            guard !state.closed else { return false }
            state.users += 1
            return true
        }
    }

    func end() {
        let completion = state.withLock { state -> (@Sendable () -> Void)? in
            state.users -= 1
            guard state.users == 0 else { return nil }
            defer { state.onDrain = nil }
            return state.onDrain
        }
        completion?()
    }

    /// Called once by the outgoing session. No new pump may enter after this point.
    func close(whenDrained completion: @escaping @Sendable () -> Void) {
        let alreadyDrained = state.withLock { state in
            state.closed = true
            guard state.users > 0 else { return true }
            state.onDrain = completion
            return false
        }
        if alreadyDrained { completion() }
    }
}
#endif
