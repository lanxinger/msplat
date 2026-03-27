import SwiftUI
import UIKit
import Msplat
import QuartzCore

// MARK: - Pixel conversion

nonisolated(unsafe) var displayReady: Int32 = 1

func pixelDataToUIImage(_ pd: PixelData) -> UIImage {
    let w = pd.width, h = pd.height
    let n = w * h
    let rgba = UnsafeMutablePointer<UInt8>.allocate(capacity: n * 4)
    pd.pixels.withUnsafeBufferPointer { src in
        for i in 0..<n {
            rgba[i * 4]     = UInt8(min(max(src[i * 3], 0), 1) * 255)
            rgba[i * 4 + 1] = UInt8(min(max(src[i * 3 + 1], 0), 1) * 255)
            rgba[i * 4 + 2] = UInt8(min(max(src[i * 3 + 2], 0), 1) * 255)
            rgba[i * 4 + 3] = 255
        }
    }
    let data = Data(bytesNoCopy: rgba, count: n * 4, deallocator: .custom { ptr, _ in
        ptr.deallocate()
    })
    let provider = CGDataProvider(data: data as CFData)!
    let cgImg = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32,
                        bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
                        bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                        provider: provider, decode: nil, shouldInterpolate: false,
                        intent: .defaultIntent)!
    return UIImage(cgImage: cgImg)
}

// MARK: - Vector helpers

func normalize(_ v: (Float, Float, Float)) -> (Float, Float, Float) {
    let len = sqrt(v.0*v.0 + v.1*v.1 + v.2*v.2)
    guard len > 1e-8 else { return (0, 1, 0) }
    return (v.0/len, v.1/len, v.2/len)
}

func cross(_ a: (Float, Float, Float), _ b: (Float, Float, Float)) -> (Float, Float, Float) {
    (a.1*b.2 - a.2*b.1, a.2*b.0 - a.0*b.2, a.0*b.1 - a.1*b.0)
}

// MARK: - Orbit camera

func lookAtCamToWorld(eye: (Float, Float, Float), target: (Float, Float, Float),
                      up: (Float, Float, Float)) -> [Float] {
    let f = normalize((eye.0 - target.0, eye.1 - target.1, eye.2 - target.2))
    let r = normalize(cross(up, f))
    let u = cross(f, r)
    return [
        r.0, u.0, f.0, eye.0,
        r.1, u.1, f.1, eye.1,
        r.2, u.2, f.2, eye.2,
          0,   0,   0,     1,
    ]
}

struct OrbitParams {
    var lookAt: (Float, Float, Float)
    var eyeCenter: (Float, Float, Float)
    var radius: Float
    var up: (Float, Float, Float)
    var tangent1: (Float, Float, Float)
    var tangent2: (Float, Float, Float)
}

func computeOrbitParams(_ poses: [[Float]]) -> OrbitParams {
    let n = Float(poses.count)
    var cx: Float = 0, cy: Float = 0, cz: Float = 0

    for p in poses {
        cx += p[3]; cy += p[7]; cz += p[11]
    }
    cx /= n; cy /= n; cz /= n

    // Use camera 0's up vector (column 1 of camToWorld) so the orbit
    // orientation matches what the user sees during training
    let p0 = poses[0]
    let up = normalize((p0[1], p0[5], p0[9]))

    var totalR: Float = 0
    for p in poses {
        let dx = p[3] - cx, dy = p[7] - cy, dz = p[11] - cz
        let dot = dx*up.0 + dy*up.1 + dz*up.2
        let gx = dx - dot*up.0, gy = dy - dot*up.1, gz = dz - dot*up.2
        totalR += sqrt(gx*gx + gy*gy + gz*gz)
    }
    let radius = totalR / n * 2.0

    let lookAt = (cx, cy, cz) as (Float, Float, Float)
    let eyeCenter = (
        cx + up.0 * radius * 0.05,
        cy + up.1 * radius * 0.05,
        cz + up.2 * radius * 0.05
    )

    let absX = abs(up.0), absY = abs(up.1), absZ = abs(up.2)
    let seed: (Float, Float, Float)
    if absX <= absY && absX <= absZ { seed = (1, 0, 0) }
    else if absY <= absZ { seed = (0, 1, 0) }
    else { seed = (0, 0, 1) }

    let t1 = normalize(cross(up, seed))
    let t2 = cross(up, t1)

    return OrbitParams(lookAt: lookAt, eyeCenter: eyeCenter, radius: radius,
                       up: up, tangent1: t1, tangent2: t2)
}

// MARK: - Engine

@MainActor
final class Engine: ObservableObject {
    @Published var image: UIImage?
    @Published var iteration: Int = 0
    @Published var totalIterations: Int = 2_000
    @Published var splatCount: Int = 0
    @Published var msPerStep: Float = 0
    @Published var fps: Float = 0
    @Published var phase: Phase = .idle
    @Published var loadingStatus: String = ""
    @Published var loadingSeconds: Int = 0
    private var loadingTimer: Timer?

    enum Phase: Equatable {
        case idle, loading, training, orbiting
    }

    private func startLoadingTimer() {
        loadingSeconds = 0
        loadingTimer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.loadingSeconds += 1 }
        }
    }

    private func stopLoadingTimer() {
        loadingTimer?.invalidate()
        loadingTimer = nil
    }

    func start(datasetURL: URL) {
        phase = .loading
        loadingStatus = "Reading dataset..."
        startLoadingTimer()

        let path = datasetURL.path
        let screenBounds = UIScreen.main.nativeBounds
        Thread.detachNewThread { [weak self, screenBounds] in
            guard let self else { return }
          autoreleasepool {

            // High downscale for iOS memory constraints — keeps all cameras for full coverage
            DispatchQueue.main.async { self.loadingStatus = "Loading images..." }
            let dataset = GaussianDataset(path: path, downscaleFactor: 16.0)

            DispatchQueue.main.async { self.loadingStatus = "Initializing model..." }
            var config = TrainingConfig()
            config.iterations = 2_000
            config.numDownscales = 0
            config.bgColor = (0, 0, 0)
            let trainer = GaussianTrainer(dataset: dataset, config: config)

            DispatchQueue.main.async {
                self.stopLoadingTimer()
                self.phase = .training
            }

            // Phase 1: Training
            var batchStart = CACurrentMediaTime()
            var batchSteps = 0

            for i in 0..<2_000 {
                let stats = trainer.step()
                batchSteps += 1

                if i % 25 == 0 || i == 1_999 {
                    let batchEnd = CACurrentMediaTime()
                    let avgMs = Float((batchEnd - batchStart) / Double(batchSteps) * 1000.0)

                    let pd = trainer.render(cameraIndex: 0)
                    let img = pixelDataToUIImage(pd)
                    let iter = stats.iteration
                    let count = stats.splatCount
                    DispatchQueue.main.async {
                        self.image = img
                        self.iteration = iter
                        self.splatCount = count
                        self.msPerStep = avgMs
                    }

                    batchStart = CACurrentMediaTime()
                    batchSteps = 0
                }
            }

            let finalCount = trainer.splatCount
            DispatchQueue.main.async {
                self.splatCount = finalCount
                self.phase = .orbiting
            }

            // Phase 2: Smooth circular orbit
            let poses = (0..<dataset.numTrain).map { dataset.cameraPose(at: $0) }
            let orbit = computeOrbitParams(poses)
            var frameCount = 0
            var fpsTimer = CACurrentMediaTime()

            // Render at half native screen resolution with wide FOV
            let renderScale: Float = 0.5
            let renderW: Int32 = Int32(Float(screenBounds.width) * renderScale)
            let renderH: Int32 = Int32(Float(screenBounds.height) * renderScale)
            let fovY: Float = 60.0 * .pi / 180.0
            var imgW: Int32 = renderW, imgH: Int32 = renderH
            let rgbaBuf = UnsafeMutablePointer<UInt8>.allocate(capacity: Int(renderW * renderH) * 4)
            for i in 0..<Int(renderW * renderH) { rgbaBuf[i * 4 + 3] = 255 }

            let colorSpace = CGColorSpaceCreateDeviceRGB()
            let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)

            let orbitPeriod: Double = 7.5
            let orbitStart = CACurrentMediaTime()

            while true {
                autoreleasepool {
                    let elapsed = CACurrentMediaTime() - orbitStart
                    let angle = Float(elapsed / orbitPeriod) * 2.0 * .pi
                    let cosA = cos(angle), sinA = sin(angle)
                    let t1 = orbit.tangent1, t2 = orbit.tangent2
                    let eye = (
                        orbit.eyeCenter.0 + orbit.radius * (cosA * t1.0 + sinA * t2.0),
                        orbit.eyeCenter.1 + orbit.radius * (cosA * t1.1 + sinA * t2.1),
                        orbit.eyeCenter.2 + orbit.radius * (cosA * t1.2 + sinA * t2.2)
                    )

                    let pose = lookAtCamToWorld(eye: eye, target: orbit.lookAt, up: orbit.up)
                    trainer.renderWithFovToBuffer(camToWorld: pose, width: renderW, height: renderH,
                                                  fovY: fovY, rgba: rgbaBuf,
                                                  outWidth: &imgW, outHeight: &imgH)
                    frameCount += 1

                    var currentFps: Float = 0
                    let now = CACurrentMediaTime()
                    let dt = now - fpsTimer
                    if dt >= 0.5 {
                        currentFps = Float(frameCount) / Float(dt)
                        frameCount = 0
                        fpsTimer = now
                    }

                    if OSAtomicCompareAndSwap32(1, 0, &displayReady) {
                        let provider = CGDataProvider(dataInfo: nil, data: rgbaBuf,
                                                       size: Int(imgW * imgH) * 4,
                                                       releaseData: { _, _, _ in })!
                        let cgImg = CGImage(width: Int(imgW), height: Int(imgH),
                                            bitsPerComponent: 8, bitsPerPixel: 32,
                                            bytesPerRow: Int(imgW) * 4, space: colorSpace,
                                            bitmapInfo: bitmapInfo, provider: provider,
                                            decode: nil, shouldInterpolate: false,
                                            intent: .defaultIntent)!
                        let img = UIImage(cgImage: cgImg)
                        let fpsVal = currentFps
                        DispatchQueue.main.async {
                            self.image = img
                            if fpsVal > 0 { self.fps = fpsVal }
                            OSAtomicCompareAndSwap32(0, 1, &displayReady)
                        }
                    }
                }
            }
          } // autoreleasepool (outer)
        }
    }
}

// MARK: - Folder picker

struct FolderPicker: UIViewControllerRepresentable {
    let onPick: (URL) -> Void

    func makeUIViewController(context: Context) -> UIDocumentPickerViewController {
        let picker = UIDocumentPickerViewController(forOpeningContentTypes: [.folder])
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ vc: UIDocumentPickerViewController, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator(onPick: onPick) }

    class Coordinator: NSObject, UIDocumentPickerDelegate {
        let onPick: (URL) -> Void
        init(onPick: @escaping (URL) -> Void) { self.onPick = onPick }

        func documentPicker(_ controller: UIDocumentPickerViewController,
                            didPickDocumentsAt urls: [URL]) {
            guard let url = urls.first else { return }
            // Start security-scoped access for the selected folder
            _ = url.startAccessingSecurityScopedResource()
            onPick(url)
        }
    }
}

// MARK: - Indeterminate progress bar

struct IndeterminateBar: View {
    @State private var offset: CGFloat = -1.0

    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width
            RoundedRectangle(cornerRadius: 2)
                .fill(.white.opacity(0.15))
                .overlay(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 2)
                        .fill(
                            LinearGradient(
                                colors: [Color(red: 0.2, green: 0.5, blue: 1.0),
                                         Color(red: 0.0, green: 0.9, blue: 0.7)],
                                startPoint: .leading, endPoint: .trailing
                            )
                        )
                        .frame(width: w * 0.35)
                        .offset(x: offset * w * 0.65)
                }
                .clipShape(RoundedRectangle(cornerRadius: 2))
        }
        .onAppear {
            withAnimation(.easeInOut(duration: 1.0).repeatForever(autoreverses: true)) {
                offset = 1.0
            }
        }
    }
}

// MARK: - UI

struct ContentView: View {
    @StateObject private var engine = Engine()
    @State private var showPicker = false

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            if let img = engine.image {
                Image(uiImage: img)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
            }

            // Idle: show pick button
            if engine.phase == .idle {
                VStack(spacing: 20) {
                    Text("msplat")
                        .font(.system(size: 36, weight: .bold, design: .monospaced))
                        .foregroundStyle(.white)
                    Text("Select a COLMAP / Nerfstudio dataset folder")
                        .foregroundStyle(.white.opacity(0.7))
                    Button {
                        showPicker = true
                    } label: {
                        Label("Open Dataset", systemImage: "folder")
                            .font(.headline)
                            .padding(.horizontal, 24)
                            .padding(.vertical, 12)
                            .background(.white.opacity(0.15))
                            .cornerRadius(12)
                    }
                    .tint(.white)
                }
            }

            if engine.phase == .loading {
                VStack(spacing: 16) {
                    IndeterminateBar()
                        .frame(width: 220, height: 4)

                    Text(engine.loadingStatus)
                        .font(.system(size: 16, weight: .medium, design: .monospaced))
                        .foregroundStyle(.white)

                    Text("\(engine.loadingSeconds)s")
                        .font(.system(size: 14, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.5))
                }
            }

            if engine.phase == .training || engine.phase == .orbiting {
                VStack(spacing: 0) {
                    Spacer()

                    if engine.phase == .training {
                        progressBar
                    }

                    statsBar
                }
                .ignoresSafeArea(.container, edges: .bottom)
            }

            // FPS counter top-right during orbit
            if engine.phase == .orbiting {
                VStack {
                    HStack {
                        Spacer()
                        Text(String(format: "%.0f FPS", engine.fps))
                            .font(.system(size: 28, weight: .bold, design: .monospaced))
                            .foregroundStyle(.white)
                            .padding(.horizontal, 16)
                            .padding(.vertical, 8)
                            .background(.black.opacity(0.7))
                            .cornerRadius(10)
                            .padding(16)
                    }
                    Spacer()
                }
            }
        }
        .sheet(isPresented: $showPicker) {
            FolderPicker { url in
                engine.start(datasetURL: url)
            }
        }
        .statusBarHidden(engine.phase == .training || engine.phase == .orbiting)
    }

    private var progressBar: some View {
        let progress = Double(engine.iteration) / Double(engine.totalIterations)
        return GeometryReader { geo in
            ZStack(alignment: .leading) {
                Rectangle()
                    .fill(.white.opacity(0.15))
                Rectangle()
                    .fill(
                        LinearGradient(
                            colors: [Color(red: 0.2, green: 0.5, blue: 1.0),
                                     Color(red: 0.0, green: 0.9, blue: 0.7)],
                            startPoint: .leading, endPoint: .trailing
                        )
                    )
                    .frame(width: geo.size.width * progress)
                    .animation(.linear(duration: 0.1), value: progress)
            }
        }
        .frame(height: 6)
    }

    private var statsBar: some View {
        HStack(spacing: 24) {
            switch engine.phase {
            case .idle, .loading:
                EmptyView()
            case .training:
                Text("step \(engine.iteration) / \(engine.totalIterations)")
                Text("\(fmtCount(engine.splatCount)) splats")
                Text(String(format: "%.1f ms/step", engine.msPerStep))
            case .orbiting:
                Text("\(fmtCount(engine.splatCount)) splats")
                Text(String(format: "%.0f fps", engine.fps))
            }
        }
        .font(.system(size: 15, weight: .semibold, design: .monospaced))
        .foregroundStyle(.white)
        .frame(maxWidth: .infinity)
        .padding(.vertical, 12)
        .padding(.bottom, 16) // extra padding for home indicator
        .background(.black.opacity(0.85))
    }

    private func fmtCount(_ n: Int) -> String {
        if n >= 1_000_000 { return String(format: "%.2fM", Double(n) / 1_000_000) }
        if n >= 1_000 { return String(format: "%.0fK", Double(n) / 1_000) }
        return "\(n)"
    }
}

// MARK: - App entry

@main
struct DemoApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .preferredColorScheme(.dark)
        }
    }
}
