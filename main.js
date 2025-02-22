const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const { PythonShell } = require('python-shell');

let mainWindow;  // Define mainWindow in a broader scope

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

  console.log('Selected file paths:', result.filePaths);

  if (result.filePaths.length > 0) {
    const videoPath = result.filePaths[0];
    console.log('Sending video path to Python script:', videoPath);

    const options = {
      args: [videoPath],
      pythonOptions: ['-u'], // Use unbuffered mode to get real-time output
    };

    const pyshell = new PythonShell('D:\\coding\\python\\coded\\my-electron-app\\model4.py', options);

    // Listen for output from the Python script
    pyshell.on('message', (message) => {
      console.log('Python script message:', message);
      
      // Extract progress percentage using a regex
      const progressRegex = /- (\d+\.\d+)% completed/;
      const match = message.match(progressRegex);
      if (match) {
        const percentage = parseFloat(match[1]);
        // Send the progress update to the renderer process
        mainWindow.webContents.send("progress-update", percentage);
      }
      
      // When processing is complete, notify the renderer
      if (message.includes('Processing complete')) {
        mainWindow.webContents.send('analysis:finished', videoPath.replace('.mp4', '_processed.mp4'));
      }
    });

    // Handle any errors
    pyshell.on('error', (err) => {
      console.error('Error running Python script:', err);
    });

    // Handle the end of the script
    pyshell.end((err) => {
      if (err) {
        console.error('Python shell ended with error:', err);
      } else {
        console.log('Python script finished successfully');
      }
    });
  }

  return result.filePaths;
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
