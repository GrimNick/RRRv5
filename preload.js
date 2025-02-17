const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("electron", {
    openFileDialog: () => ipcRenderer.invoke("dialog:openFile"),
    startAnalysis: (filePath) => ipcRenderer.send("start-analysis", filePath),
    onProgressUpdate: (callback) => {
        const listener = (event, percent) => callback(percent);
        ipcRenderer.on("progress-update", listener);
        return () => ipcRenderer.removeListener("progress-update", listener);
    },
    onAnalysisComplete: (callback) => {
        const listener = (event, outputPath) => callback(outputPath);
        ipcRenderer.on("analysis-complete", listener);
        return () => ipcRenderer.removeListener("analysis-complete", listener);
    },
    onAnalysisError: (callback) => {
        const listener = (event, errorMessage) => callback(errorMessage);
        ipcRenderer.on("analysis-error", listener);
        return () => ipcRenderer.removeListener("analysis-error", listener);
    },
    openExcelFile: (filePath) => {
        if (!filePath) {
            console.error("ERROR: No file path provided for Excel file.");
            return;
        }
        ipcRenderer.send("open-excel", filePath);
    },

    removeListeners: () => {
        ipcRenderer.removeAllListeners("progress-update");
        ipcRenderer.removeAllListeners("analysis-complete");
        ipcRenderer.removeAllListeners("analysis-error");
        ipcRenderer.removeAllListeners("exit-app");  
    },

    exitApp: () => {
        console.log("Calling `exitApp` from preload.js...");
        ipcRenderer.send("exit-app");  
    }
});

console.log(" Preload.js Loaded - API Exposed");  
