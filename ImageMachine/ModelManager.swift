import Foundation
import CoreML
import StableDiffusion
import ZIPFoundation
import os

/// Owns the on-device Stable Diffusion model: getting it onto the phone once,
/// then loading it for fully offline generation.
@MainActor
final class ModelManager: ObservableObject {
    enum State: Equatable {
        case missing
        case downloading(Double)
        case unzipping
        case loading
        case ready
        case failed(String)
    }

    /// The two models the app can use. Only one is kept on the phone at a time.
    enum Choice: String, CaseIterable, Identifiable {
        /// Apple's 4-bit SDXL split for iPhone's Neural Engine: 768x768, best quality.
        case sdxl
        /// Apple's 6-bit Stable Diffusion 2.1: 512x512, faster, fits 6 GB phones.
        case sd21

        var id: String { rawValue }

        var title: String {
            switch self {
            case .sdxl: return "SDXL · 768×768 · best quality"
            case .sd21: return "SD 2.1 · 512×512 · fastest"
            }
        }

        var size: String {
            switch self {
            case .sdxl: return "about 2 to 3 GB"
            case .sd21: return "about 1.5 GB"
            }
        }

        /// This is the only time the app touches the network.
        var downloadURL: URL {
            switch self {
            case .sdxl: return URL(string: "https://huggingface.co/apple/coreml-stable-diffusion-xl-base-ios/resolve/main/coreml-stable-diffusion-xl-base-ios_split_einsum_compiled.zip")!
            case .sd21: return URL(string: "https://huggingface.co/apple/coreml-stable-diffusion-2-1-base-palettized/resolve/main/coreml-stable-diffusion-2-1-base-palettized_split_einsum_v2_compiled.zip")!
            }
        }

        /// SDXL wants 8 GB of RAM (iPhone 15 Pro, 16, 16 Pro and later).
        static var recommended: Choice { ModelManager.hasLotsOfMemory ? .sdxl : .sd21 }
    }

    /// iPhone 15 Pro and iPhone 16 Pro Max report about 8.0 GB here.
    nonisolated static let hasLotsOfMemory = ProcessInfo.processInfo.physicalMemory > 7_900_000_000

    @Published private(set) var state: State = .missing
    private(set) var pipeline: PipelineBox?

    private let modelsDir: URL = {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent("Models", isDirectory: true)
    }()

    init() {
        if Self.findModelFolder(in: modelsDir) != nil {
            Task { await load() }
        }
    }

    // MARK: Getting the model

    func download(_ choice: Choice) async {
        do {
            try prepareModelsDir()
            state = .downloading(0)
            let zip = try await Downloader.download(choice.downloadURL) { [weak self] fraction in
                Task { @MainActor in self?.state = .downloading(fraction) }
            }
            try await unzip(zip, deleteAfter: true)
            await load()
        } catch {
            state = .failed("Download failed: \(error.localizedDescription)")
        }
    }

    /// Lets someone AirDrop or copy the model .zip into Files and import it, with no network at all.
    func importZip(from url: URL) async {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        do {
            try prepareModelsDir()
            try await unzip(url, deleteAfter: false)
            await load()
        } catch {
            state = .failed("Import failed: \(error.localizedDescription)")
        }
    }

    func deleteModel() {
        pipeline = nil
        try? FileManager.default.removeItem(at: modelsDir)
        state = .missing
    }

    private func prepareModelsDir() throws {
        try FileManager.default.createDirectory(at: modelsDir, withIntermediateDirectories: true)
        // Keep gigabytes of model weights out of iCloud backups.
        var dir = modelsDir
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try dir.setResourceValues(values)
    }

    private func unzip(_ zip: URL, deleteAfter: Bool) async throws {
        state = .unzipping
        let dest = modelsDir
        try await Task.detached(priority: .userInitiated) {
            try FileManager.default.unzipItem(at: zip, to: dest)
            if deleteAfter { try? FileManager.default.removeItem(at: zip) }
        }.value
    }

    // MARK: Loading

    func load() async {
        guard let folder = Self.findModelFolder(in: modelsDir) else {
            state = .failed("Couldn't find the model files after unpacking. Try deleting and downloading again.")
            return
        }
        state = .loading
        let isXL = FileManager.default.fileExists(atPath: folder.appendingPathComponent("TextEncoder2.mlmodelc").path)
        do {
            pipeline = try await Task.detached(priority: .userInitiated) {
                let config = MLModelConfiguration()
                // The models are split_einsum builds, laid out for the Neural Engine;
                // keeping the GPU out avoids shuffling weights between the two.
                config.computeUnits = .cpuAndNeuralEngine
                // Keep every part of the model resident when iOS gives us room for it
                // (8 GB phones with the increased-memory-limit entitlement), so nothing
                // reloads between images. Otherwise load each part only while it's needed.
                let needed: UInt64 = isXL ? 4_000_000_000 : 2_500_000_000
                let reduceMemory = UInt64(os_proc_available_memory()) < needed
                let pipeline: StableDiffusionPipelineProtocol
                if isXL {
                    pipeline = try StableDiffusionXLPipeline(
                        resourcesAt: folder,
                        configuration: config,
                        reduceMemory: reduceMemory
                    )
                } else {
                    pipeline = try StableDiffusionPipeline(
                        resourcesAt: folder,
                        controlNet: [],
                        configuration: config,
                        // No safety checker: it isn't loaded, so it costs no memory or time.
                        disableSafety: true,
                        reduceMemory: reduceMemory
                    )
                }
                // Loading also compiles the model for this phone's Neural Engine,
                // which iOS caches, so only the first launch is slow.
                try pipeline.loadResources()
                return PipelineBox(pipeline, isXL: isXL)
            }.value
            state = .ready
        } catch {
            state = .failed("Couldn't load the model: \(error.localizedDescription)")
        }
    }

    /// The zip unpacks into a folder whose name we don't control, so look for the
    /// directory that actually holds the compiled models.
    nonisolated static func findModelFolder(in root: URL) -> URL? {
        let fm = FileManager.default
        guard let walker = fm.enumerator(at: root, includingPropertiesForKeys: [.isDirectoryKey]) else { return nil }
        for case let url as URL in walker where url.lastPathComponent == "TextEncoder.mlmodelc" {
            let folder = url.deletingLastPathComponent()
            let hasUnet = fm.fileExists(atPath: folder.appendingPathComponent("Unet.mlmodelc").path)
                || fm.fileExists(atPath: folder.appendingPathComponent("UnetChunk1.mlmodelc").path)
            if hasUnet { return folder }
        }
        return nil
    }
}

/// The pipelines aren't Sendable; we only ever use one from one task at a time.
final class PipelineBox: @unchecked Sendable {
    let pipeline: StableDiffusionPipelineProtocol
    let isXL: Bool
    init(_ pipeline: StableDiffusionPipelineProtocol, isXL: Bool) {
        self.pipeline = pipeline
        self.isXL = isXL
    }
}

/// Minimal download helper that reports progress for a large file.
final class Downloader: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let onProgress: (Double) -> Void
    private var continuation: CheckedContinuation<URL, Error>?

    private init(onProgress: @escaping (Double) -> Void) {
        self.onProgress = onProgress
    }

    static func download(_ url: URL, onProgress: @escaping (Double) -> Void) async throws -> URL {
        let delegate = Downloader(onProgress: onProgress)
        let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }
        return try await withCheckedThrowingContinuation { cont in
            delegate.continuation = cont
            session.downloadTask(with: url).resume()
        }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64, totalBytesWritten: Int64,
                    totalBytesExpectedToWrite: Int64) {
        guard totalBytesExpectedToWrite > 0 else { return }
        onProgress(Double(totalBytesWritten) / Double(totalBytesExpectedToWrite))
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didFinishDownloadingTo location: URL) {
        // The temp file disappears when this method returns, so move it now.
        let kept = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".zip")
        do {
            if let http = downloadTask.response as? HTTPURLResponse, http.statusCode != 200 {
                throw URLError(.badServerResponse)
            }
            try FileManager.default.moveItem(at: location, to: kept)
            continuation?.resume(returning: kept)
        } catch {
            continuation?.resume(throwing: error)
        }
        continuation = nil
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error {
            continuation?.resume(throwing: error)
            continuation = nil
        }
    }
}
