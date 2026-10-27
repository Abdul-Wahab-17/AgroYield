// Base URL of the Flask API. Set REACT_APP_API_URL at build time (e.g. in Vercel)
// to point the deployed frontend at the hosted API.
export const API_URL = (process.env.REACT_APP_API_URL || 'http://localhost:5000').replace(/\/$/, '');
