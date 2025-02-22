const { app, BrowserWindow, ipcMain, dialog } = require("electron");
const path = require("path");
const fs = require("fs");
const { PythonShell } = require("python-shell");
const { exec } = require("child_process");

let mainWindow;
let excelFilePath1, excelFilePath2, excelFilePath3;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 800,
    height: 800,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
  });

  mainWindow.loadFile("index.html");

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

ipcMain.handle("dialog:openFile", async (event) => {
  const result = await dialog.showOpenDialog({
    properties: ["openFile"],
    filters: [{ name: "Videos", extensions: ["mp4", "avi", "mkv"] }],
  });

  if (result.filePaths.length > 0) {
    const videoPath = result.filePaths[0];
    excelFilePath1 = videoPath.replace(".mp4", "_processed_data.xlsx");
    excelFilePath2 = videoPath.replace(".mp4", "_processed_data5.xlsx");
    excelFilePath3 = videoPath.replace(".mp4", "_processed_data6.xlsx");

    console.log("Running Python script for:", videoPath);

    const pythonScriptPath = path.join(__dirname, "model4.py");

    const options = {
      args: [videoPath],
      pythonPath: process.platform === "darwin" ? "python3" : "python",
      pythonOptions: ["-u"],
    };

    const pyshell = new PythonShell(pythonScriptPath, options);

    pyshell.on("message", (message) => {
      console.log("Python Output:", message);

      const progressMatch = message.match(/- (\d+\.\d+)% completed/);
      if (progressMatch) {
        const percentage = parseFloat(progressMatch[1]);
        mainWindow.webContents.send("progress-update", percentage);
      }

      if (message.includes("Processing complete")) {
        mainWindow.webContents.send(
          "analysis:finished",
          videoPath.replace(".mp4", "_processed.mp4")
        );
      }
    });

    pyshell.on("error", (err) => {
      console.error("Python script error:", err);
    });

    pyshell.end((err) => {
      if (err) console.error("Python shell ended with error:", err);
      else console.log("Python script execution completed.");
    });
  }

  return result.filePaths;
});

app.whenReady().then(() => {
  createWindow();

  function openExcelFile(filePath) {
    if (filePath && fs.existsSync(filePath)) {
      const openCommand = process.platform === "darwin" ? `open "${filePath}"` : `start "" "${filePath}"`;
      exec(openCommand, (err) => {
        if (err) console.error(`Failed to open ${filePath}:`, err);
        else console.log(`${filePath} opened successfully`);
      });
    } else {
      console.log("Excel file not found:", filePath);
    }
  }

  ipcMain.on("open-excel-file1", () => openExcelFile(excelFilePath1));
  ipcMain.on("open-excel-file2", () => openExcelFile(excelFilePath2));
  ipcMain.on("open-excel-file3", () => openExcelFile(excelFilePath3));

  // Exit App when requested from Renderer
  ipcMain.on("app:exit", () => {
    console.log("Exit request received. Closing application...");
    if (mainWindow) mainWindow.close();
    setTimeout(() => app.quit(), 500);
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    console.log("All windows closed. Quitting app...");
    app.quit();
  }
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});
