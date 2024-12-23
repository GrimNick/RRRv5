document.getElementById('uploadButton').addEventListener('click', function () {
    document.getElementById('fileInput').click();
});

document.getElementById('fileInput').addEventListener('change', function (event) {
    if (event.target.files.length > 0) {
        alert(`Video selected: ${event.target.files[0].name}`);
    }
});
