import React from "react";
import spinner from "./mole.gif";

export default () => {
  return (
    <div>
      <img
        src={spinner}
        style={{ width: "700px", margin: "auto", display: "block" }}
        alt="Loading..."
      />
    </div>
  );
};
