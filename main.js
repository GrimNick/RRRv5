// const { app, BrowserWindow, ipcMain, dialog } = require('electron');
// const path = require('path');
// const { PythonShell } = require('python-shell');

// let mainWindow;

// function createWindow() {
//   mainWindow = new BrowserWindow({
//     width: 800,
//     height: 800,
//     webPreferences: {
//       nodeIntegration: true,
//       contextIsolation: false,
//     },
//   });

//   mainWindow.loadFile('index.html');
// }

// ipcMain.handle('dialog:openFile', async (event) => {
//   const result = await dialog.showOpenDialog({
//     properties: ['openFile'],
//     filters: [
//       { name: 'Videos', extensions: ['mp4', 'avi', 'mkv'] },
//     ],
//   });

//   if (result.filePaths.length > 0) {
//     const videoPath = result.filePaths[0];
//     const scriptFile = path.join(__dirname, 'model4.py'); // Adjusted path

//     const options = {
//       args: [videoPath],
//       pythonOptions: ['-u'],
//     };

//     const pyshell = new PythonShell(scriptFile, options);

//     pyshell.on('message', (message) => {
//       if (message.includes('Processing complete')) {
//         const processedVideoPath = videoPath.replace('.mp4', '_processed.mp4');
//         mainWindow.webContents.send('analysis:finished', processedVideoPath);
//       }
//     });

//     pyshell.on('error', (err) => {
//       console.error('Error running Python script:', err);
//     });

//     pyshell.end((err) => {
//       if (err) console.error('Python shell ended with error:', err);
//     });
//   }

//   return result.filePaths;
// });

// app.whenReady().then(createWindow);

// app.on('window-all-closed', () => {
//   if (process.platform !== 'darwin') {
//     app.quit();
//   }
// });

// app.on('activate', () => {
//   if (BrowserWindow.getAllWindows().length === 0) {
//     createWindow();
//   }
// });
// ﻿



// **Handle App Closing**
const { app, BrowserWindow, ipcMain, dialog, shell } = require("electron");
const path = require("path");
const { spawn } = require("child_process");
const fs = require("fs");

let mainWindow;

app.whenReady().then(() => {
    mainWindow = new BrowserWindow({
        width: 1200,
        height: 800,
        webPreferences: {
            preload: path.join(__dirname, "preload.js"), 
            contextIsolation: true,
            enableRemoteModule: false,
            nodeIntegration: false,
        }
    });

    mainWindow.loadFile(path.join(__dirname, "index.html"));
    mainWindow.webContents.openDevTools(); 
    mainWindow.on("close", () => {
        console.log("🚪 Window closed, quitting app...");
        app.quit();
    });
});

//  File Selection Dialog for MP4 Videos
ipcMain.handle("dialog:openFile", async () => {
    const result = await dialog.showOpenDialog({
        properties: ["openFile"],
        filters: [{ name: "Videos", extensions: ["mp4", "avi", "mov"] }],
    });

    return result.filePaths.length > 0 ? result.filePaths[0] : null;
});

// Start Video Analysis
ipcMain.on("start-analysis", (event, filePath) => {
    if (!filePath || !fs.existsSync(filePath)) {
        console.error(" Error: File does not exist:", filePath);
        event.reply("analysis-error", "File not found.");
        return;
    }

    console.log("Starting analysis for:", filePath);
    let pythonProcess = spawn("python3", ["modelVideoAnalysis.py", filePath]);

    let excelFilePath = null; // Store Excel file path

    pythonProcess.stdout.on("data", (data) => {
        let output = data.toString().trim();
        console.log("[PYTHON OUTPUT]:", output);

        //  Capture Progress Updates
        let match = output.match(/\[INFO\] Progress: (\d+)%/);
        if (match) {
            let progress = parseInt(match[1]);
            console.log("Progress Updated:", progress, "%");
            event.reply("progress-update", progress);
        }

        //  Capture Excel File Path from Python Output
        let excelMatch = output.match(/Excel file created at:\s*(.+\.xlsx)/);
        if (excelMatch) {
            excelFilePath = excelMatch[1].trim();
            console.log(" Excel File Detected:", excelFilePath);
            event.reply("analysis-complete", excelFilePath); //  Notify renderer immediately
        }
    });

    pythonProcess.stderr.on("data", (data) => {
        let errorMsg = data.toString().trim();
        console.error(" [PYTHON ERROR]:", errorMsg);
        event.reply("analysis-error", `Python Error: ${errorMsg}`);
    });

    pythonProcess.on("exit", (code) => {
        if (code === 0 && excelFilePath) {
            console.log("Analysis completed successfully.");
        } else {
            console.error(" Python script exited with error code:", code);
            event.reply("analysis-error", `Analysis failed. Exit code: ${code}`);
        }
    });
});

//  Open Excel File Automatically
ipcMain.on("open-excel", (event, filePath) => {
    if (!filePath || !fs.existsSync(filePath)) {
        console.error(" Excel file not found:", filePath);
        event.reply("analysis-error", "Excel file not found.");
        return;
    }

    console.log("Opening Excel File:", filePath);
    shell.openPath(filePath);
});

//  Handle App Exit (Final Fix for Exit Button)
ipcMain.on("exit-app", () => {
    console.log("Exit-app event received. Closing application...");

    if (mainWindow) {
        mainWindow.close(); //  Closes the window before quitting
    }

    setTimeout(() => {
        app.quit();
    }, 500); //  Small delay to allow cleanup
});
