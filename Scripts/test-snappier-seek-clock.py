"""Run the ported software clock paths and upstream audio policy tests without building the engine."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root / 'Sources/AetherEngine/Native/SoftwarePlaybackHost.swift').read_text()


def block(signature):
    start = source.index(signature)
    opening = source.index('{', start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


harness = '''import Foundation
enum EngineLog {
    enum Category { case swPlayback }
    static func emit(_ message: String, category: Category) {}
}
final class ClockOutput {
    var currentTimeSeconds = 100.0
    var rate: Float = 1
    func pause() { rate = 0 }
    func setRate(_ value: Float) { rate = value }
}
final class SoftwareClockHarness {
    var isLive = false
    var decoupleAudio = true
    var audioOutput: ClockOutput? = ClockOutput()
    var playing = true
    var armed = true
    var pauseArmOpen = false
    var rebuffering = false
    var lastEnqueuedAudioPtsSec = Double.nan
    var diagAudioPtsSec = Double.nan
    var diagAudioPtsGeneration: UInt64 = 0
    var parkedSeekGeneration: UInt64 = 0
    var publishedBuffering: [Bool] = []
    func clockArmed() -> Bool { armed }
    func isPlaying() -> Bool { playing }
    func currentRate() -> Float { playing ? 1 : 0 }
    func freeParkedVideo() {}
    func noteRebuffering(_ value: Bool) { publishedBuffering.append(value) }
    CLOCK_ACTION
    func completeSeek(generation gen: UInt64 = 1) {
        SEEK_RESET
    }
}
'''.replace('CLOCK_ACTION', block('func applyAudioClockAction()')).replace('SEEK_RESET', block('if gen != parkedSeekGeneration'))

with tempfile.TemporaryDirectory(prefix='snappier-clock-tests-') as directory:
    temp = Path(directory)
    sources = temp / 'Sources/AetherEngine'
    tests = temp / 'Tests/AetherEngineTests'
    sources.mkdir(parents=True)
    tests.mkdir(parents=True)
    (temp / 'Package.swift').write_text('''// swift-tools-version: 6.2
import PackageDescription
let package = Package(name: "SnappierClock", platforms: [.macOS(.v15)], targets: [
    .target(name: "AetherEngine"),
    .testTarget(name: "AetherEngineTests", dependencies: ["AetherEngine"])
])
''')
    shutil.copy(root / 'Sources/AetherEngine/Native/AudioLookahead.swift', sources)
    (sources / 'SoftwareClockHarness.swift').write_text(harness)
    shutil.copy(root / 'Tests/AetherEngineTests/AudioLookaheadTests.swift', tests)
    shutil.copy(root / 'Scripts/tests/SnappierSeekClockTests.swift', tests)
    subprocess.run(['swift', 'test', '--package-path', str(temp), '--disable-sandbox'], check=True,
                   env=dict(os.environ, CLANG_MODULE_CACHE_PATH=str(temp / 'ModuleCache'),
                            SWIFTPM_MODULECACHE_OVERRIDE=str(temp / 'ModuleCache')))
