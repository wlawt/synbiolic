const express = require("express");
const mongoose = require("mongoose");
const bodyParser = require("body-parser");
const passport = require("passport");
const path = require("path");
const axios = require("axios");
var util = require("util");

const molecules = require("./routes/api/molecules/molecule");
const users = require("./routes/api/users/user");

const app = express();

// Body parser middleware
app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());

// DB Config
const keys = require("./config/keys_dev");
const db = keys.mongoURI;

// Connect to MongoDB
mongoose
  .connect(db, { useNewUrlParser: true, useUnifiedTopology: true })
  .then(() => console.log("MongoDB Connected"))
  .catch(err => console.log(err));

// Passport middleware
app.use(passport.initialize());

// Passport config
require("./config/passport")(passport);

// API endpoints
app.use("/api/molecules/molecule", molecules);
app.use("/api/users/user", users);

/* 
  SERVER SIDE REQ
*/

// Generate molecule - AZURE
var r;
app.post("/gen", (req, res) => {
  axios
    .post(`${keys.azure}`, req.body, {
      headers: {
        Authorization: "Bearer " + `${keys.token}`,
        "Content-Type": "application/json"
      }
    })
    .then(re => {
      r = JSON.parse(JSON.stringify(util.inspect(re.data)));

      var a_smiles = [];
      for (var i = 0; i < re.data.smiles.length; i++) {
        a_smiles.push(re.data.smiles[i]);
      }

      var a_pic50 = [];
      for (var i = 0; i < re.data.pic50.length; i++) {
        a_pic50.push(re.data.pic50[i]);
      }

      var a_qed = [];
      for (var i = 0; i < re.data.qed.length; i++) {
        a_qed.push(re.data.qed[i]);
      }

      var a_pic_dist = [];
      for (var i = 0; i < re.data.pic50_dist.length; i++) {
        a_pic_dist.push(re.data.pic50_dist[i]);
      }

      const result = {
        smiles: a_smiles,
        pic50: a_pic50,
        qed: a_qed,
        pic_dist: a_pic_dist,
        n_generated: parseInt(util.inspect(re.data.n_generated))
      };

      //console.log(result);
      return res.json(result);
    })
    .catch(err => console.log(err));
});

// Predictions - IBM
app.post("/pred_retro", (req, res) => {
  //console.log(typeof req.body); -> object
  //console.log(req.body);
  axios
    .post(`${keys.ibm}`, req.body, {
      headers: {
        Authorization: `${keys.ibm_token}`,
        "Content-Type": "application/json",
        Accept: "application/json"
      }
    })
    .then(re => {
      var s = JSON.parse(JSON.stringify(util.inspect(re.data.payload.id)));

      //console.log(s);
      return res.json(s);
    })
    .catch(err => console.log(err));
});

// Retrosynthesis - IBM
app.get("/get_retro", (req, res) => {
  axios
    .get(`https://rxn.res.ibm.com/rxn/api/api/v1/retrosynthesis/${req.body.id}`)
    .then(re => {
      return res.json(util.inspect(re));
    })
    .catch(err => console.log(err));
});

// Server static assets if in production
if (process.env.NODE_ENV === "production") {
  // Set static folder
  app.use(express.static("client/build"));

  app.get("*", (req, res) => {
    res.sendFile(path.resolve(__dirname, "client", "build", "index.html"));
  });
}

const port = process.env.PORT || 5000;

app.listen(port, () => console.log(`Server running on port ${port}`));
