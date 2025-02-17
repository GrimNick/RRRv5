document.addEventListener("DOMContentLoaded", async () => {
    console.log(" DOM Loaded - Initializing event listeners");
    console.log("Current Page:", window.location.pathname); // Debugging

    const uploadButton = document.getElementById("uploadButton");
    const statusDiv = document.getElementById("status");
    const progressContainer = document.getElementById("progressContainer");
    const progressBar = document.getElementById("progressBar");
    const displayAnalysisButton = document.getElementById("displayAnalysisButton");
    const displayListButton = document.getElementById("displayRecklessListButton");
    const returnHomeButton = document.getElementById("returnHome");
    const exitAppButton = document.getElementById("exitApp");

    //  Handle Video Upload - ONLY on `index.html`
    if (window.location.pathname.includes("index.html")) {
        console.log(" Detected index.html page");

        if (uploadButton) {
            uploadButton.addEventListener("click", async () => {
                console.log(" Select Video button clicked!");
                uploadButton.innerText = "Please Wait...";
                uploadButton.disabled = true;
                statusDiv.innerText = "Processing selected file...";
                progressContainer.style.display = "block";
                progressBar.style.width = "0%";
                progressBar.innerText = "0%";

                try {
                    const filePath = await window.electron.openFileDialog();
                    if (!filePath) {
                        console.warn("⚠ No file selected.");
                        statusDiv.innerText = "⚠ No file selected.";
                        resetUI();
                        return;
                    }

                    console.log("Selected File:", filePath);
                    statusDiv.innerText = "Processing...";
                    sessionStorage.setItem("videoPath", filePath);
                    window.electron.startAnalysis(filePath);
                    window.electron.removeListeners();

                    const removeProgressListener = window.electron.onProgressUpdate((percent) => {
                        percent = Math.min(100, Math.max(0, percent));
                        progressBar.style.width = `${percent}%`;
                        progressBar.innerText = `${percent}%`;
                        console.log(` Progress Updated: ${percent}%`);
                    });

                    const removeCompletionListener = window.electron.onAnalysisComplete((outputPath) => {
                        console.log("Analysis Completed! Output:", outputPath);
                        statusDiv.innerText = "";
                        uploadButton.innerText = "Finished Analysis";
                        uploadButton.disabled = true;
                        progressBar.style.width = "100%";
                        progressBar.innerText = "100%";

                        setTimeout(() => {
                            progressContainer.style.display = "none";
                            uploadButton.style.display = "none";
                            displayAnalysisButton.style.display = "block";
                        }, 2000);

                        if (outputPath.endsWith(".mp4") || outputPath.endsWith(".avi") || outputPath.endsWith(".mov")) {
                            sessionStorage.setItem("videoPath", outputPath);
                        } else if (outputPath.endsWith(".xlsx")) {
                            sessionStorage.setItem("excelFilePath", outputPath);
                        }

                        displayAnalysisButton.onclick = () => {
                            const videoFile = sessionStorage.getItem("videoPath");
                            if (videoFile) {
                                window.location.href = `playvideo.html?videoPath=${encodeURIComponent(videoFile)}`;
                            } else {
                                alert("⚠ No valid video file found.");
                            }
                        };

                        removeProgressListener();
                        removeCompletionListener();
                    });

                    const removeErrorListener = window.electron.onAnalysisError((errorMessage) => {
                        console.error("Analysis Error:", errorMessage);
                        statusDiv.innerText = `Error: ${errorMessage}`;
                        resetUI();
                        removeProgressListener();
                        removeCompletionListener();
                        removeErrorListener();
                    });

                } catch (error) {
                    console.error(" File selection error:", error);
                    statusDiv.innerText = " Error selecting file.";
                    resetUI();
                }
            });
        } else {
            console.warn("⚠ Select Video Button Not Found.");
        }
    }

    // Handle Display Reckless List - ONLY on `playvideo.html`
    if (window.location.pathname.includes("playvideo.html")) {
        console.log(" Detected playvideo.html page");

        if (displayListButton) {
            displayListButton.style.display = "block";
            displayListButton.addEventListener("click", () => {
                const excelFilePath = sessionStorage.getItem("excelFilePath");
                if (excelFilePath) {
                    window.electron.openExcelFile(excelFilePath);
                } else {
                    alert("⚠ No reckless list available yet. Run analysis first.");
                }
            });
        }
    } else {
        if (displayListButton) displayListButton.style.display = "none";
    }

    //  Handle Home & Exit Buttons - ONLY on `playvideo.html`
    if (window.location.pathname.includes("playvideo.html")) {
        console.log(" Running Home & Exit Button logic on playvideo.html");

        if (returnHomeButton) {
            console.log("Home Button Found. Adding event...");
            returnHomeButton.onclick = () => {
                console.log("Returning to Home Page...");
                window.location.href = "index.html";
            };
        } else {
            console.warn("⚠ Home Button Not Found. (Skipping event listener)");
        }

        if (exitAppButton) {
            console.log("Exit Button Found. Adding event...");
            exitAppButton.onclick = () => {
                console.log(" Exit button clicked! Calling exitApp()...");
                window.electron.exitApp();
            };
        } else {
            console.warn("⚠ Exit Button Not Found. (Skipping event listener)");
        }
    }

    function resetUI() {
        if (uploadButton) {
            uploadButton.innerText = "Select Video";
            uploadButton.disabled = false;
        }
        if (progressContainer) {
            progressContainer.style.display = "none";
        }
        if (progressBar) {
            progressBar.style.width = "0%";
        }
        if (displayAnalysisButton) {
            displayAnalysisButton.style.display = "none";
        }
    }
});
