## MLP Architecture

| Layer  | Size | Filter | Activation |
|--------|------|--------|------------|
| Dense  | 128  | --     | relu       |
| Dense  | 64   | --     | relu       |
| Dense  | N    | --     | softmax    |

## CNN Architecture

| Layer       | Size | Filter | Activation |
|------------|------|--------|------------|
| Convolution | 128  | 3 × 3  | relu       |
| Max Pooling | 2    | --     | --         |
| Convolution | 64   | 3 × 3  | relu       |
| Max Pooling | 2    | --     | --         |
| Convolution | 32   | 3 × 3  | relu       |
| Max Pooling | 2    | --     | --         |
| Flatten     | --   | 3 × 3  | --         |
| Dense       | 128  | --     | relu       |
| Dense       | 64   | --     | relu       |
| Dense       | N    | --     | softmax    |

## RNN Architecture

| Layer | Size | Filter | Activation |
|-------|------|--------|------------|
| LSTM  | 128  | --     | --         |
| Dense | 64   | --     | relu       |
| Dense | N    | --     | softmax    |
