import {
  GET_DATA,
  LOADING,
  GET_RETRO,
  GET_RETRO_ID,
  SAVE_STATS,
  GET_STATS,
  SAVE_RETRO_STATS,
  GET_RETRO_STATS
} from "../actions/types";

const initialState = {
  molecules: [],
  retros: [],
  retro_res: [],
  loading: false,
  saved: [],
  retro_stats: []
};

export default function(state = initialState, action) {
  switch (action.type) {
    case LOADING:
      return {
        ...state,
        loading: true
      };
    case GET_DATA:
      return {
        ...state,
        molecules: action.payload,
        loading: false
      };
    case GET_RETRO_ID:
      return {
        ...state,
        retros: action.payload,
        loading: false
      };
    case GET_RETRO:
      return {
        ...state,
        retro_res: action.payload,
        loading: false
      };
    case SAVE_STATS:
      return {
        ...state,
        saved: [action.payload, ...state.saved]
      };
    case GET_STATS:
      return {
        ...state,
        saved: action.payload,
        loading: false
      };
    case SAVE_RETRO_STATS:
      return {
        ...state,
        retro_stats: [action.payload, ...state.retro_stats]
      };
    case GET_RETRO_STATS:
      return {
        ...state,
        retro_stats: action.payload,
        loading: false
      };
    default:
      return state;
  }
}
