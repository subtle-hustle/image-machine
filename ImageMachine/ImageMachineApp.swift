import SwiftUI

@main
struct ImageMachineApp: App {
    @StateObject private var model = ModelManager()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(model)
        }
    }
}
