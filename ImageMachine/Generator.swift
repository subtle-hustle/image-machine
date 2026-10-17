import Foundation
import CoreGraphics
import StableDiffusion

/// Runs one text-to-image job at a time on the loaded pipeline.
@MainActor
final class Generator: ObservableObject {
    @Published var prompt = ""
    @Published var negativePrompt = "blurry, low quality, distorted"
    @Published var steps: Double = 20
    @Published var guidance: Double = 7.5
    @Published var seed: UInt32? = nil

    @Published private(set) var isRunning = false
    @Published private(set) var progress: Double = 0
    @Published private(set) var image: CGImage?
    @Published private(set) var lastSeed: UInt32?
    @Published private(set) var message: String?

    func generate(with box: PipelineBox) {
        guard !isRunning, !prompt.trimmingCharacters(in: .whitespaces).isEmpty else { return }

        var config = StableDiffusionPipeline.Configuration(prompt: prompt)
        config.negativePrompt = negativePrompt
        config.stepCount = Int(steps)
        config.guidanceScale = Float(guidance)
        config.seed = seed ?? UInt32.random(in: 0...UInt32.max)
        config.schedulerType = .dpmSolverMultistepScheduler
        config.disableSafety = true
        if box.isXL {
            // SDXL's VAE uses a different latent scale, and its sampler does
            // better with Karras spacing at low step counts.
            config.encoderScaleFactor = 0.13025
            config.decoderScaleFactor = 0.13025
            config.schedulerTimestepSpacing = .karras
        }

        isRunning = true
        cancelledFlag.set(false)
        progress = 0
        message = nil
        lastSeed = config.seed

        Task.detached(priority: .userInitiated) { [weak self] in
            do {
                let images = try box.pipeline.generateImages(configuration: config) { p in
                    let fraction = Double(p.step + 1) / Double(p.stepCount)
                    Task { @MainActor in self?.progress = fraction }
                    return !(self?.isCancelledSnapshot ?? false)
                }
                let first = images.first ?? nil
                await self?.finish(image: first)
            } catch {
                await self?.finish(error: error)
            }
        }
    }

    func cancel() { cancelledFlag.set(true) }

    // The progress callback runs off the main actor, so it reads a thread-safe copy.
    private let cancelledFlag = AtomicFlag()
    nonisolated var isCancelledSnapshot: Bool { cancelledFlag.get() }

    private func finish(image: CGImage?) {
        isRunning = false
        if cancelledFlag.get() {
            message = "Stopped."
        } else if let image {
            self.image = image
        } else {
            message = "No image came back. Try again."
        }
    }

    private func finish(error: Error) {
        isRunning = false
        message = "Generation failed: \(error.localizedDescription)"
    }
}

final class AtomicFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false
    func get() -> Bool { lock.lock(); defer { lock.unlock() }; return value }
    func set(_ v: Bool) { lock.lock(); value = v; lock.unlock() }
}
