let express = require("express");
let router = express.Router();
let path = require("path");
let fs = require("fs");

let rootPath = path.join(__dirname, "..");

function sendError(message) {
    let error = {
        msg: message,
        status: 400
    };

    return error;
}

router.get("/:dataset/:type", function(req, res, next) {
    let dirFiles;
    dataPath = path.join(rootPath, "data/", req.params.dataset);

    if (!fs.existsSync(dataPath)) {
        res.json(sendError("File not found"));
        return;
    }

    dirFiles = fs.readdirSync(dataPath);
    dirFiles.forEach(file => {
        if (file.includes(req.params.type))
            res.sendFile(path.join(dataPath, file));
    });
});

module.exports = router;