import React from "react";
import Logo from "../img/logo.png";

const Footer = () => {
  return (
    <footer className="container py-3">
      <div className="row">
        <div className="col-12 col-md">
          <div>
            <img src={Logo} width="200" height="200" alt="" />
          </div>
          <small className="d-block mb-3 text-muted">
            &copy; {new Date().getFullYear()}
          </small>
        </div>
        {/* <div class="col-6 col-md">
          <h5>Features</h5>
          <ul class="list-unstyled text-small">
            <li>
              <a class="text-muted" href="#">
                Cool stuff
              </a>
            </li>
            <li>
              <a class="text-muted" href="#">
                Random feature
              </a>
            </li>
            <li>
              <a class="text-muted" href="#">
                Team feature
              </a>
            </li>
            <li>
              <a class="text-muted" href="#">
                Stuff for developers
              </a>
            </li>
            <li>
              <a class="text-muted" href="#">
                Another one
              </a>
            </li>
            <li>
              <a class="text-muted" href="#">
                Last time
              </a>
            </li>
          </ul>
        </div>
        <div class="col-6 col-md">
          <h5>Resources</h5>
          <ul class="list-unstyled text-small">
            <li>
              <a class="text-muted" href="#">
                Resource
              </a>
            </li>
            <li>
              <a class="text-muted" href="#">
                Resource name
              </a>
            </li>
            <li>
              <a class="text-muted" href="#">
                Another resource
              </a>
            </li>
            <li>
              <a class="text-muted" href="#">
                Final resource
              </a>
            </li>
          </ul>
        </div>
        <div class="col-6 col-md">
          <h5>Resources</h5>
          <ul class="list-unstyled text-small">
            <li>
              <a class="text-muted" href="#">
                Business
              </a>
            </li>
            <li>
              <a class="text-muted" href="#">
                Education
              </a>
            </li>
            <li>
              <a class="text-muted" href="#">
                Government
              </a>
            </li>
            <li>
              <a class="text-muted" href="#">
                Gaming
              </a>
            </li>
          </ul>
        </div>
        <div class="col-6 col-md">
          <h5>About</h5>
          <ul class="list-unstyled text-small">
            <li>
              <a class="text-muted" href="#">
                Team
              </a>
            </li>
            <li>
              <a class="text-muted" href="#">
                Locations
              </a>
            </li>
            <li>
              <a class="text-muted" href="#">
                Privacy
              </a>
            </li>
            <li>
              <a class="text-muted" href="#">
                Terms
              </a>
            </li>
          </ul>
        </div> */}
      </div>
    </footer>
  );
};

export default Footer;
