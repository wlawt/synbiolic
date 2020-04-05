const Validator = require("validator");
const isEmpty = require("./is-empty");

module.exports = function validateTemplateInput(data) {
  let errors = {};

  // Make sure empty fields are converted to strings
  data.template = !isEmpty(data.template) ? data.template : "";

  if (Validator.isEmpty(data.template)) {
    errors.template = "Template field is required";
  }

  return {
    errors,
    isValid: isEmpty(errors)
  };
};
