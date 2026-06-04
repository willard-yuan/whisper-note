import SwiftUI

struct CompanionChatView: View {
    @State private var vm = CompanionChatViewModel()
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                if !vm.modelsLoaded {
                    loadingSection
                } else {
                    chatList

                    statusBar

                    if vm.isListening {
                        DiagnosticsView(monitor: vm.diagnostics, asrBackend: vm.asrBackend)

                        VoiceLevelBar(level: vm.audioLevel)
                            .frame(height: 4)
                            .padding(.horizontal)
                    }

                    inputBar
                }
            }
            .navigationTitle("Transcription")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .onChange(of: scenePhase) { _, newPhase in
                switch newPhase {
                case .background:
                    // Stop pipeline to free audio resources
                    if vm.isListening { vm.stopListening() }
                case .active:
                    // If models were unloaded (OOM kill), reload
                    if !vm.modelsLoaded && !vm.isLoading {
                        Task { await vm.loadModels() }
                    }
                default:
                    break
                }
            }
            .toolbar {
                ToolbarItem(placement: .automatic) {
                    if vm.modelsLoaded {
                        Menu {
                            Button("Clear Chat") { vm.clearChat() }
                            Divider()
                            if vm.isListening {
                                Button("Stop Listening") { vm.stopListening() }
                            } else {
                                Button("Start Listening") { vm.startListening() }
                            }
                        } label: {
                            Image(systemName: "ellipsis.circle")
                        }
                    }
                }
            }
        }
    }

    // MARK: - Loading

    private var loadingSection: some View {
        VStack(spacing: 16) {
            Spacer()

            if vm.isLoading {
                ProgressView(value: vm.loadProgress) {
                    Text(vm.loadingStatus)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal, 40)
            } else {
                Image(systemName: "bubble.left.and.bubble.right.fill")
                    .font(.system(size: 48))
                    .foregroundStyle(.tint)

                Text("Transcription")
                    .font(.title2.bold())

                Text("On-device Qwen3 ASR\nVAD + pseudo-streaming")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)

                Button("Load Models") {
                    Task { await vm.loadModels() }
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
            }

            if let error = vm.errorMessage {
                Text(error)
                    .foregroundStyle(.red)
                    .font(.caption)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)

                Button("Retry") {
                    Task { await vm.loadModels() }
                }
                .buttonStyle(.bordered)
            }

            Spacer()
        }
    }

    // MARK: - Chat

    private var chatList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 8) {
                    ForEach(vm.messages) { msg in
                        ChatBubble(message: msg)
                            .id(msg.id)
                    }

                    if !vm.currentPartial.isEmpty {
                        partialBubble(vm.currentPartial)
                            .id("partial")
                    }

                }
                .padding(.vertical, 8)
            }
            .onChange(of: vm.messages.count) {
                withAnimation {
                    if let last = vm.messages.last {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
            }
            .onChange(of: vm.currentPartial) {
                if !vm.currentPartial.isEmpty {
                    withAnimation {
                        proxy.scrollTo("partial", anchor: .bottom)
                    }
                }
            }
        }
    }

    private func partialBubble(_ text: String) -> some View {
        HStack {
            Spacer(minLength: 60)

            Text(text)
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .background(Color.accentColor.opacity(0.55))
                .foregroundStyle(.white)
                .clipShape(RoundedRectangle(cornerRadius: 18))
                .overlay(alignment: .bottomTrailing) {
                    Image(systemName: "waveform")
                        .font(.caption2)
                        .foregroundStyle(.white.opacity(0.85))
                        .padding(6)
                }
        }
        .padding(.horizontal)
    }

    // MARK: - Status Bar

    private var statusBar: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(statusColor)
                .frame(width: 8, height: 8)

            Text(vm.pipelineState)
                .font(.caption2)
                .foregroundStyle(.secondary)

            Spacer()

            if vm.isSpeechDetected {
                Image(systemName: "waveform")
                    .foregroundStyle(.red)
                    .font(.caption)
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 4)
    }

    private var statusColor: Color {
        if vm.isSpeechDetected { return .red }
        if vm.isListening { return .green }
        return .gray
    }

    // MARK: - Input (text fallback)

    private var inputBar: some View {
        HStack(spacing: 8) {
            Button {
                if vm.isListening {
                    vm.stopListening()
                } else {
                    vm.startListening()
                }
            } label: {
                Image(systemName: vm.isListening ? "mic.fill" : "mic.slash")
                    .font(.title3)
                    .foregroundStyle(vm.isSpeechDetected ? .red : (vm.isListening ? .green : .gray))
            }

            TextField("Add note...", text: $vm.inputText, axis: .vertical)
                .textFieldStyle(.plain)
                .lineLimit(1...4)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color.gray.opacity(0.15))
                .clipShape(RoundedRectangle(cornerRadius: 20))
                .onSubmit { sendIfReady() }

            Button {
                sendIfReady()
            } label: {
                Image(systemName: "arrow.up.circle.fill")
                    .font(.title2)
            }
            .disabled(
                vm.inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            )
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
        .background(.bar)
    }

    private func sendIfReady() {
        let text = vm.inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        vm.send(text)
    }
}
