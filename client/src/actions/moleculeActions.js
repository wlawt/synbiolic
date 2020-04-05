import axios from "axios";
import {
  GET_DATA,
  LOADING,
  GET_RETRO,
  GET_RETRO_ID,
  SAVE_STATS,
  GET_ERRORS,
  GET_STATS,
  SAVE_RETRO_STATS,
  GET_RETRO_STATS
} from "./types";

export const generate_molecules = data => dispatch => {
  dispatch(set_loading());
  axios
    .post("/gen", data)
    .then(res =>
      dispatch({
        type: GET_DATA,
        payload: res
      })
    )
    .catch(err =>
      dispatch({
        type: GET_DATA,
        payload: null
      })
    );
};

export const retrosynthesis_molecule = data => dispatch => {
  dispatch(set_loading());
  axios
    .post("/pred_retro", data)
    .then(res =>
      dispatch({
        type: GET_RETRO_ID,
        payload: res
      })
    )
    .catch(err =>
      dispatch({
        type: GET_RETRO_ID,
        payload: null
      })
    );
};

export const get_retro_molecule = id => dispatch => {
  dispatch(set_loading());
  axios
    .get("/get_retro", id)
    .then(res =>
      dispatch({
        type: GET_RETRO,
        payload: res
      })
    )
    .catch(err =>
      dispatch({
        type: GET_RETRO,
        payload: null
      })
    );
};

export const save_molecule_stats = stats => dispatch => {
  axios
    .post("/api/molecules/molecule/save", stats)
    .then(res =>
      dispatch({
        type: SAVE_STATS,
        payload: res.data
      })
    )
    .catch(err =>
      dispatch({
        type: GET_ERRORS,
        payload: null
      })
    );
};

export const save_retro_stats = stats => dispatch => {
  axios
    .post("/api/molecules/molecule/save-retro", stats)
    .then(res =>
      dispatch({
        type: SAVE_RETRO_STATS,
        payload: res.data
      })
    )
    .catch(err =>
      dispatch({
        type: GET_ERRORS,
        payload: null
      })
    );
};

export const get_molecule_stats = () => dispatch => {
  dispatch(set_loading());
  axios
    .get("/api/molecules/molecule")
    .then(res =>
      dispatch({
        type: GET_STATS,
        payload: res
      })
    )
    .catch(err =>
      dispatch({
        type: GET_STATS,
        payload: null
      })
    );
};

export const get_retro_stats = () => dispatch => {
  dispatch(set_loading());
  axios
    .get("/api/molecules/molecule/retro")
    .then(res =>
      dispatch({
        type: GET_RETRO_STATS,
        payload: res
      })
    )
    .catch(err =>
      dispatch({
        type: GET_RETRO_STATS,
        payload: null
      })
    );
};

export const set_loading = () => {
  return {
    type: LOADING
  };
};
