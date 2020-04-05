const express = require("express");
const router = express.Router();

const Molecule = require("../../../models/Molecule");
const Retro = require("../../../models/Retro");

/*  @route      POST api/molecules/molecule/save
    @desc       Save data
    @access     public
*/
router.post("/save", (req, res) => {
  //console.log(req.body);
  const newStat = new Molecule({
    molecule: req.body.molecule,
    pic50: req.body.pic50,
    qed: req.body.qed
  });

  newStat.save().then(stat => res.json(stat));
});

/*  @route      GET api/molecules/molecule
    @desc       Get data
    @access     public
*/
router.get("/", (req, res) => {
  Molecule.find()
    .then(molecules => res.json(molecules))
    .catch(err => res.status(404).json({ nomolecules: "No molecules found!" }));
});

/*  @route      POST api/molecules/molecule/save-retro
    @desc       Save retro data
    @access     public
*/
router.post("/save-retro", (req, res) => {
  //console.log(req.body);
  const newStat = new Retro({
    molecule: req.body.molecule,
    pic50: req.body.pic50,
    qed: req.body.qed
  });

  newStat.save().then(stat => res.json(stat));
});

/*  @route      GET api/molecules/molecule/retro
    @desc       Get data
    @access     public
*/
router.get("/retro", (req, res) => {
  Retro.find()
    .then(retros => res.json(retros))
    .catch(err =>
      res.status(404).json({ noretros: "No retrosynthesis stats saved!" })
    );
});

module.exports = router;
