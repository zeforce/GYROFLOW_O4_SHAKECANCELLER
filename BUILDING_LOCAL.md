# Windows build

This fork is based on Gyroflow commit `4632d48`. Build it with the same toolchain and SDKs required by upstream Gyroflow: Rust/Cargo, Visual Studio C++ Build Tools, Qt 6.7, FFmpeg, OpenCV, LLVM/libclang, and MDK SDK. See the upstream [build instructions](https://github.com/gyroflow/gyroflow#building-gyroflow).

A release build produces `gyroflow.exe` in Cargo's release target directory. To distribute it, include the matching Qt, FFmpeg, OpenCV, MDK and C++ runtime DLLs and plugins, as in the Release ZIP. Keep the executable name `Gyroflow.exe` because the included MDK license identifies the application by name.

The O4 repair implementation is in `src/core/gyro_source/o4_repair.rs`. The GUI is in `src/ui/menu/MotionData.qml`, `src/ui/components/Timeline.qml`, and `src/ui/VideoArea.qml`. Simplified Chinese translations are in `resources/translations/zh_CN.ts` and its compiled `.qm` file.
