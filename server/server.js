let express = require("express");
var path = require('path');
var cookieParser = require('cookie-parser');
let app = express();

let data = require("./routes/data");

app.set('views', path.join(__dirname, 'views'));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(cookieParser());

app.use("/data", data);

console.log("Running on port");

app.listen(20793);