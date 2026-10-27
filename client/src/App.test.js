import { render, screen } from '@testing-library/react';
import App from './App';

beforeEach(() => {
  global.fetch = jest.fn(() =>
    Promise.resolve({
      ok: true,
      json: () => Promise.resolve({ crops: ['CORN'], states: ['IOWA'], available_models: ['Random Forest'] }),
    })
  );
});

test('renders the navigation bar', () => {
  render(<App />);
  expect(screen.getByText(/Crop Yield Predictor/i)).toBeInTheDocument();
  expect(screen.getByRole('link', { name: 'Predict' })).toBeInTheDocument();
  expect(screen.getByRole('link', { name: 'Metrics' })).toBeInTheDocument();
});
