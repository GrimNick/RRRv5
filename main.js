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


const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const { PythonShell } = require('python-shell');

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 800,
    height: 800,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
  });

  mainWindow.loadFile('index.html');
}

ipcMain.handle('dialog:openFile', async (event) => {
  const result = await dialog.showOpenDialog({
    properties: ['openFile'],
    filters: [
      { name: 'Videos', extensions: ['mp4', 'avi', 'mkv'] },
    ],
  });

  if (result.filePaths.length > 0) {
    const videoPath = result.filePaths[0];
    const scriptFile = path.join(__dirname, 'model4.py'); // Adjusted path

    const options = {
      args: [videoPath],
      pythonOptions: ['-u'],
    };

    const pyshell = new PythonShell(scriptFile, options);

    pyshell.on('message', (message) => {
      if (message.includes('Processing complete')) {
        const processedVideoPath = videoPath.replace('.mp4', '_processed.mp4');
        mainWindow.webContents.send('analysis:finished', processedVideoPath);
      }
    });

    pyshell.on('error', (err) => {
      console.error('Error running Python script:', err);
    });

    pyshell.end((err) => {
      if (err) console.error('Python shell ended with error:', err);
    });
  }

  return result.filePaths;
});

// Handle the exit-app event
ipcMain.on('close-app', () => {
  app.quit();
});

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});
