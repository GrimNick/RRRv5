const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const fs = require('fs');
const { PythonShell } = require('python-shell');
const { exec } = require('child_process');

let mainWindow;  // Define mainWindow in a broader scope
let excelFilePath1;
let excelFilePath2;
let excelFilePath3;

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
    excelFilePath1 = videoPath.replace('.mp4', '_processed_data.xlsx');
    excelFilePath2 = videoPath.replace('.mp4', '_processed_data5.xlsx');
    excelFilePath3 = videoPath.replace('.mp4', '_processed_data3.xlsx');

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

//app.whenReady().then(createWindow);
app.whenReady().then(() => {
  createWindow();

  // Open Excel file when requested from the renderer
  ipcMain.on('open-excel-file1', () => {
      if (fs.existsSync(excelFilePath1)) {
          // Open the Excel file using the default system application
          exec(`start "" "${excelFilePath1}"`, (err) => {
              if (err) {
                  console.error('Failed to open Excel file:', err);
              } else {
                  console.log('Excel file opened successfully');
              }
          });
      } else {
          console.log('Excel file not found');
      }
  });
  ipcMain.on('open-excel-file2', () => {

    // Check if the Excel file exists
    if (fs.existsSync(excelFilePath2)) {
        // Open the Excel file using the default system application
        exec(`start "" "${excelFilePath2}"`, (err) => {
            if (err) {
                console.error('Failed to open Excel file:', err);
            } else {
                console.log('Excel file opened successfully');
            }
        });
    } else {
        console.log('Excel file not found');
    }
});
ipcMain.on('open-excel-file3', () => {

  // Check if the Excel file exists
  if (fs.existsSync(excelFilePath3)) {
      // Open the Excel file using the default system application
      exec(`start "" "${excelFilePath3}"`, (err) => {
          if (err) {
              console.error('Failed to open Excel file:', err);
          } else {
              console.log('Excel file opened successfully');
          }
      });
  } else {
      console.log('Excel file not found');
  }
});




});

//delete till here if not working newwwwtttonnn
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
