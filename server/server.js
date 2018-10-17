let express = require("express");
var path = require("path");
var cookieParser = require("cookie-parser");
let serverless = require("serverless-http");
let app = express();

let index = require("./routes/index");
let data = require("./routes/data");

app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'ejs');
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(cookieParser());

app.use(express.static(path.join(__dirname, 'public')));

app.use("/", index);
app.use("/data", data);

console.log("Running on port");

app.listen(20793);

module.exports.handler = serverless(app);