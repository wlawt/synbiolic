import React, { Fragment } from "react";
import classnames from "classnames";
import PropTypes from "prop-types";

const InputGroup = ({
  name,
  placeholder,
  value,
  error,
  onChange,
  type,
  html_for,
  label_desc
}) => {
  return (
    <Fragment>
      {/*       <label className="text-left font-weight-bold" htmlFor={html_for}>
        {label_desc}
      </label> */}
      <div className="input-group mb-3 w-75">
        <input
          className={classnames("form-control form-control-lg", {
            "is-invalid": error
          })}
          placeholder={placeholder}
          value={value}
          onChange={onChange}
          name={name}
        />
        {error && <div className="invalid-feedback">{error}</div>}
      </div>
    </Fragment>
  );
};

InputGroup.propTypes = {
  name: PropTypes.string.isRequired,
  placeholder: PropTypes.string,
  value: PropTypes.string.isRequired,
  type: PropTypes.string.isRequired,
  error: PropTypes.string,
  onChange: PropTypes.func.isRequired,
  html_for: PropTypes.string,
  label_desc: PropTypes.string
};

InputGroup.defaultProps = {
  type: "text"
};

export default InputGroup;
