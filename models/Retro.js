const mongoose = require("mongoose");
const Schema = mongoose.Schema;

const RetroSchema = new Schema({
  molecule: {
    type: String,
    required: true
  },
  pic50: {
    type: String,
    required: true
  },
  qed: {
    type: String,
    required: true
  }
});

module.exports = Retro = mongoose.model("retros", RetroSchema);
