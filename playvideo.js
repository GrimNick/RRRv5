document.addEventListener("DOMContentLoaded", () => {
    console.log("DOM Loaded - Initializing video player and event listeners");

    const videoPlayer = document.getElementById("videoPlayer");
    const recklessListButton = document.getElementById("displayRecklessListButton");
    const returnHomeButton = document.getElementById("returnHome");
    const exitAppButton = document.getElementById("exitApp");

    //  Get video path from URL or sessionStorage
    const urlParams = new URLSearchParams(window.location.search);
    let videoPath = urlParams.get("videoPath") || sessionStorage.getItem("videoPath");

    if (videoPath) {
        videoPath = decodeURIComponent(videoPath);

        //  Allow only video files (MP4, AVI, MOV)
        const validExtensions = [".mp4", ".avi", ".mov"];
        const isValidVideo = validExtensions.some(ext => videoPath.toLowerCase().endsWith(ext));

        if (isValidVideo) {
            console.log(" Video Loaded:", videoPath);
            videoPlayer.src = videoPath;
            videoPlayer.style.display = "block"; // Ensure the video player is visible
            videoPlayer.load();
            videoPlayer.play();
        } else {
            console.error("Invalid file detected:", videoPath);
            alert("⚠ Error: Only video files are allowed.");
            videoPlayer.style.display = "none"; // Hide the video player if invalid
            sessionStorage.removeItem("videoPath"); // Remove invalid entry
        }
    } else {
        console.error(" No video path provided.");
        alert("⚠ Error: No video found to play.");
        videoPlayer.style.display = "none";
    }

    // Ensure "Display Reckless List" button appears only when an Excel file is available
    if (recklessListButton) {
        const excelFilePath = sessionStorage.getItem("excelFilePath");

        if (excelFilePath && excelFilePath.endsWith(".xlsx")) {
            recklessListButton.style.display = "block"; // Show the button if Excel exists
            recklessListButton.onclick = () => {
                console.log("Opening Reckless List:", excelFilePath);
                window.electron.openExcelFile(excelFilePath);
            };
        } else {
            console.warn("⚠ No Excel file available.");
            recklessListButton.style.display = "none"; // Hide if no Excel file
            sessionStorage.removeItem("excelFilePath"); // Remove invalid entry
        }
    } else {
        console.error("Reckless List button not found.");
    }

    //  Handle "Return Home" button click
    if (returnHomeButton) {
        returnHomeButton.onclick = () => {
            console.log(" Returning to Home");
            window.location.href = "index.html";
        };
    } else {
        console.error(" Return Home button not found.");
    }

    // Handle "Exit App" button click
    if (exitAppButton) {
        console.log(" Exit Button Found. Attaching event...");
        exitAppButton.addEventListener("click", () => {
            console.log(" Exit button clicked! Calling exitApp()...");
            window.electron.exitApp();
        });
    } else {
        console.error("Exit Button Not Found.");
    }
});
    