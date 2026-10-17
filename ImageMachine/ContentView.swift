import SwiftUI
import UIKit
import UniformTypeIdentifiers

struct ContentView: View {
    @EnvironmentObject private var model: ModelManager

    var body: some View {
        NavigationStack {
            Group {
                if model.state == .ready, let box = model.pipeline {
                    GenerateView(box: box)
                } else {
                    SetupView()
                }
            }
            .navigationTitle("Image Machine")
        }
    }
}

// MARK: - One-time setup

struct SetupView: View {
    @EnvironmentObject private var model: ModelManager
    @State private var importing = false
    @State private var choice = ModelManager.Choice.recommended

    var body: some View {
        VStack(spacing: 24) {
            Image(systemName: "sparkles.rectangle.stack")
                .font(.system(size: 64))
                .foregroundStyle(.tint)

            switch model.state {
            case .missing, .failed:
                Text("Image Machine makes pictures entirely on your iPhone using its Neural Engine. It needs a one-time model download, then works with no internet at all.")
                    .multilineTextAlignment(.center)
                Picker("Model", selection: $choice) {
                    ForEach(ModelManager.Choice.allCases) { option in
                        Text(option.title).tag(option)
                    }
                }
                .pickerStyle(.inline)
                .labelsHidden()
                if choice == .sdxl && !ModelManager.hasLotsOfMemory {
                    Text("SDXL needs a phone with 8 GB of RAM, like iPhone 15 Pro or any iPhone 16. It may close on this one.")
                        .font(.footnote).foregroundStyle(.orange).multilineTextAlignment(.center)
                }
                if case .failed(let reason) = model.state {
                    Text(reason).font(.footnote).foregroundStyle(.red).multilineTextAlignment(.center)
                }
                Button("Download \(choice.size) (Wi-Fi recommended)") {
                    Task { await model.download(choice) }
                }
                .buttonStyle(.borderedProminent)
                Button("Import model .zip from Files") { importing = true }
                    .font(.footnote)
            case .downloading(let fraction):
                ProgressView(value: fraction) {
                    Text("Downloading model… keep the app open")
                } currentValueLabel: {
                    Text(fraction, format: .percent.precision(.fractionLength(0)))
                }
            case .unzipping:
                ProgressView("Unpacking model…")
            case .loading:
                ProgressView("Loading model onto the Neural Engine… the first time can take a few minutes")
                    .multilineTextAlignment(.center)
            case .ready:
                ProgressView()
            }
        }
        .padding(32)
        .fileImporter(isPresented: $importing, allowedContentTypes: [.zip]) { result in
            if case .success(let url) = result {
                Task { await model.importZip(from: url) }
            }
        }
    }
}

// MARK: - Generating

struct GenerateView: View {
    let box: PipelineBox
    @EnvironmentObject private var model: ModelManager
    @StateObject private var gen = Generator()
    @State private var seedText = ""
    @State private var savedNote: String?
    @FocusState private var promptFocused: Bool

    var body: some View {
        Form {
            Section {
                ZStack {
                    RoundedRectangle(cornerRadius: 12).fill(.quaternary)
                    if let image = gen.image {
                        Image(decorative: image, scale: 1)
                            .resizable()
                            .scaledToFit()
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                    } else if !gen.isRunning {
                        Text("Your image appears here").foregroundStyle(.secondary)
                    }
                    if gen.isRunning {
                        VStack {
                            ProgressView(value: gen.progress)
                            Text("Step \(Int(gen.progress * gen.steps)) of \(Int(gen.steps))")
                                .font(.caption)
                        }
                        .padding()
                        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 8))
                        .padding()
                    }
                }
                .aspectRatio(1, contentMode: .fit)
                .listRowInsets(EdgeInsets())

                if let image = gen.image, !gen.isRunning {
                    HStack {
                        Button("Save to Photos") {
                            UIImageWriteToSavedPhotosAlbum(UIImage(cgImage: image), nil, nil, nil)
                            savedNote = "Saved."
                        }
                        Spacer()
                        ShareLink(item: Image(decorative: image, scale: 1),
                                  preview: SharePreview(gen.prompt, image: Image(decorative: image, scale: 1)))
                    }
                    .buttonStyle(.borderless)
                }
                if let note = gen.message ?? savedNote {
                    Text(note).font(.footnote).foregroundStyle(.secondary)
                }
            }

            Section("Prompt") {
                TextField("A lighthouse on a cliff at sunset, oil painting", text: $gen.prompt, axis: .vertical)
                    .lineLimit(2...5)
                    .focused($promptFocused)
                TextField("Things to avoid", text: $gen.negativePrompt, axis: .vertical)
                    .font(.footnote)
            }

            Section {
                if gen.isRunning {
                    Button("Stop", role: .destructive) { gen.cancel() }
                } else {
                    Button("Generate") {
                        promptFocused = false
                        savedNote = nil
                        gen.seed = UInt32(seedText)
                        gen.generate(with: box)
                    }
                    .disabled(gen.prompt.trimmingCharacters(in: .whitespaces).isEmpty)
                    .bold()
                }
            }

            Section("Settings") {
                VStack(alignment: .leading) {
                    Text("Steps: \(Int(gen.steps))  (more = finer detail, slower)")
                    Slider(value: $gen.steps, in: 10...50, step: 1)
                }
                VStack(alignment: .leading) {
                    Text("Prompt strength: \(gen.guidance, specifier: "%.1f")")
                    Slider(value: $gen.guidance, in: 1...15, step: 0.5)
                }
                HStack {
                    TextField("Seed (blank = random)", text: $seedText)
                        .keyboardType(.numberPad)
                    if let last = gen.lastSeed {
                        Button("Reuse \(String(last))") { seedText = String(last) }
                            .font(.caption)
                            .buttonStyle(.borderless)
                    }
                }
            }
            .disabled(gen.isRunning)

            Section {
                Button("Delete model to switch or free space", role: .destructive) { model.deleteModel() }
                    .disabled(gen.isRunning)
            } footer: {
                Text("Everything runs on this device. No prompts or images ever leave your phone.")
            }
        }
    }
}
