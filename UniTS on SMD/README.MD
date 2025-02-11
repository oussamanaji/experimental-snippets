# MIDAS: Multivariate Industrial Data Analysis System

## Quick Start

### Google Colab Notebooks
- Training Pipeline: [UniTS on SMD Training](https://colab.research.google.com/drive/1Yi1865rOT80AGRy_FXlv9Flm185GD5Km)
- Interactive Analysis Tool: [MIDAS Analysis Interface](https://colab.research.google.com/drive/1ZQeqUSfb4Bh3d5W1dTgBDDH9HiLwMF-J)

### Setup Instructions

1. **Training Pipeline Setup**
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install required dependencies
!pip install einops==0.4.0 matplotlib==3.7.0 patool==1.12 reformer-pytorch==1.4.4 
!pip install scikit-learn==1.2.2 sktime==0.16.1 sympy==1.12 tqdm==4.64.1
!pip install pyyaml wandb gluonts
!pip install --upgrade pandas
!pip install "dask[dataframe]"

# Clone UniTS repository
!rm -rf UniTS  # Remove if exists
!git clone https://github.com/mims-harvard/UniTS.git
```

2. **Analysis Tool Setup**
```python
# Install required packages
!pip install streamlit openai plotly pyngrok

# Set OpenRouter API key
import os
os.environ['OPENROUTER_API_KEY'] = 'your_api_key_here'

# Configure ngrok (optional, for public access)
from pyngrok import ngrok, conf
conf.get_default().auth_token = "your_ngrok_token"  # Optional
```

3. **Dataset Structure**
Create the following directory structure:
```
dataset/
└── SMD/
    ├── train.txt
    ├── test.txt
    └── test_label.txt
```

### Basic Usage

1. **Training the Model**
```python
# Run the training notebook
# Key configurations in Config class:
config = Config()
model = AnomalyUniTS(config)
pipeline = TrainingPipeline(model, train_loader, val_loader, test_loader, config)
best_f1 = pipeline.train()
```

2. **Using the Analysis Interface**
```python
# Run the analysis interface
!streamlit run app.py

# Example queries:
- "Analyze bearing temperature and pump motor load patterns during the last 30 minutes"
- "Check for early warning signs in coolant pressure and flow rate"
- "What caused the spike in pump vibration levels during peak load?"
```

## 1. Introduction

### Project Overview
MIDAS (Multivariate Industrial Data Analysis System) is a comprehensive anomaly detection and analysis system that combines the power of the UniTS (Universal Time Series) framework with specialized nuclear industry applications. The project consists of two main components:
- A training pipeline implementing multivariate anomaly detection
- An interactive analysis tool with nuclear industry-specific visualizations and safety monitoring

### Objectives
- Implement multivariate anomaly detection for complex industrial systems
- Provide real-time monitoring and early warning capabilities
- Enable root cause analysis through correlation studies
- Ensure compliance with nuclear industry safety standards
- Offer intuitive natural language interaction for system analysis

### Problem Statement
Industrial systems, particularly in nuclear applications, generate multivariate time series data that requires continuous monitoring for anomalous behavior. The challenge lies in:
- Detecting anomalies across multiple correlated parameters
- Identifying root causes of system deviations
- Providing early warning of potential issues
- Maintaining compliance with safety regulations

## 2. Background

### 2.1 Server Machine Dataset (SMD)
The Server Machine Dataset (SMD) serves as the foundation for this project:

#### Dataset Characteristics
- 5-week collection period from a large Internet company
- Data from 28 server machines
- 38 metrics per machine including:
  - CPU utilization
  - Memory usage
  - Network traffic
  - Disk I/O
  - System load

#### Data Structure
- Training and testing sets for each machine
- Labeled anomalies in test set
- Dimension indicators for anomaly contributions

#### Adaptation for Nuclear Applications
The SMD dataset's multivariate nature makes it suitable for nuclear applications through parameter mapping:
```python
NUCLEAR_DISPLAY_MAPPINGS = {
    'cpu_utilization': 'pump_motor_load',
    'memory_usage': 'coolant_pressure',
    'disk_io': 'flow_rate',
    # ... additional mappings
}
```

### 2.2 UniTS Framework
The Universal Time Series (UniTS) framework provides the foundation for the anomaly detection system.

#### Key Components
- Transformer-based architecture
- Multi-head attention mechanism
- Patch-based processing
- Prompt-based learning integration

## 3. Model Implementation

### 3.1 Training Pipeline
The training pipeline leverages the UniTS framework with specific adaptations for anomaly detection.

#### Configuration
```python
class Config:
    # Model architecture
    d_model = 128
    n_heads = 8
    e_layers = 3
    input_dim = 38
    seq_len = 100
    dropout = 0.1

    # Training settings
    learning_rate = 1e-4
    weight_decay = 1e-5
    epochs = 30
    batch_size = 32
    patience = 5
```

#### Data Processing
```python
class DataProcessor:
    def __init__(self, win_size=100, step=1):
        self.win_size = win_size
        self.step = step
        self.scaler = StandardScaler()

    def load_data(self, mode="train"):
        # Load and preprocess SMD dataset
        # Standardize features
        # Create sliding windows
```

#### Training Pipeline Components
1. **Data Loading and Preprocessing**
   - Standardization using StandardScaler
   - Sliding window creation
   - Train/validation split

2. **Model Architecture**
   ```python
   class AnomalyUniTS:
       def __init__(self, config):
           self.model = Model(config, task_configs)
           self.optimizer = torch.optim.Adam(
               self.model.parameters(),
               lr=config.learning_rate,
               weight_decay=config.weight_decay
           )
   ```

3. **Training Process**
   - Batch processing
   - Loss calculation
   - Early stopping
   - Model checkpointing
   - Performance monitoring

### 3.2 Anomaly Detection System

#### Detection Methodology
The system employs a reconstruction-based approach:
1. Model learns normal behavior patterns
2. Reconstruction error indicates anomalies
3. Dynamic thresholding for anomaly classification

#### Root Cause Analysis
```python
def detect_root_cause(self, reconstruction_error):
    # Average over sequence length
    feature_errors = torch.mean(reconstruction_error, dim=1)
    # Identify top contributing features
    top_k_features = torch.topk(feature_errors, k, dim=-1)
```

#### Early Warning System
```python
def early_warning_detection(self, reconstruction_error):
    # Exponential moving average for smoothing
    alpha = 0.1
    smoothed_error = torch.zeros_like(reconstruction_error)
    # Generate warning signals based on trends
```

#### Performance Metrics
The system tracks multiple performance indicators:
- Precision, Recall, F1 Score
- Detection Delay
- False Alarm Rate
- Early Warning Accuracy

## 4. Nuclear Industry Adaptation

### 4.1 Parameter Mapping

#### SMD to Nuclear Parameter Translation
The system maps server metrics to nuclear pump parameters:

```python
NUCLEAR_DISPLAY_MAPPINGS = {
    'cpu_utilization': 'pump_motor_load',
    'memory_usage': 'coolant_pressure',
    'disk_io': 'flow_rate',
    'network_incoming_traffic': 'primary_loop_flow',
    'network_outgoing_traffic': 'secondary_loop_flow',
    'cpu_temperature': 'bearing_temperature',
    # ... additional mappings
}
```

#### Component Categories
```python
component_categories = {
    'primary': [
        'pump_motor_load',
        'coolant_pressure',
        'bearing_temperature',
        'flow_rate'
    ],
    'secondary': [
        'cooling_fan_speed',
        'pump_power_draw',
        'valve_status'
    ],
    'monitoring': [
        'active_safety_systems',
        'monitoring_points',
        'system_faults'
    ]
}
```

#### Safety Thresholds
```python
SAFETY_PARAMETERS = {
    'bearing_temperature': {
        'warning': 82,
        'critical': 85,
        'reference': 'ASME Section XI',
        'description': 'Bearing temperature threshold'
    },
    'pump_motor_load': {
        'warning': 85,
        'critical': 90,
        'reference': 'IEEE 741-2007',
        'description': 'Motor load threshold'
    },
    'coolant_pressure': {
        'warning': 15.7,
        'critical': 16.0,
        'reference': 'ASME Section III',
        'description': 'Pressure threshold'
    }
}
```

### 4.2 Regulatory Compliance

#### Standards Integration
- ASME Section XI (Inservice Inspection Requirements)
- IEEE 741-2007 (Power Plant Safety Systems)
- INPO Guidelines (Preventive Maintenance)

#### Compliance Monitoring
```python
def _check_compliance(self, data: torch.Tensor, analysis_type: str) -> Dict:
    """Check regulatory compliance"""
    compliance = {}
    data_np = data.cpu().numpy()

    for component, params in self.safety_params.items():
        if 'reference' in params:
            idx = list(self.nuclear_mappings.values()).index(component)
            current = np.mean(data_np[:, idx])
            
            compliance[params['reference']] = {
                'component': component,
                'status': 'compliant' if current < params['warning'] else 'non-compliant',
                'margin': float((params['warning'] - current) / params['warning'])
            }
```

#### Safety Parameter Monitoring
- Real-time threshold monitoring
- Safety margin calculations
- Automatic alert generation
- Compliance status tracking

### 4.3 Analysis Protocols

#### Analysis Message Sequences
```python
ANALYSIS_MESSAGES = {
    'detection': [
        "Initializing pump monitoring system...",
        "Consulting ASME Section XI guidelines...",
        "Processing operational parameters...",
        # ...
    ],
    'root_cause': [
        "Initiating diagnostic sequence...",
        "Consulting maintenance specifications...",
        "Processing component correlations...",
        # ...
    ],
    'early_warning': [
        "Initializing predictive analysis...",
        "Consulting preventive maintenance standards...",
        "Processing trend indicators...",
        # ...
    ]
}
```
## 5. Visualization and Analysis Tool

### 5.1 System Architecture

#### Component Integration
```python
class SystemMonitor:
    def __init__(self):
        self.model_handler = ModelHandler()
        self.query_processor = QueryProcessor()
        self.viz_manager = VisualizationManager()
        self.data_processor = DataProcessor()
```

#### Data Flow
1. Natural language query input
2. Query processing through Claude API
3. Data analysis and processing
4. Visualization generation
5. Insight computation

#### Query Processing System
```python
class QueryProcessor:
    def process_query(self, user_query: str) -> Dict[Any, Any]:
        """Convert natural language to structured analysis"""
        completion = self.client.chat.completions.create(
            model="anthropic/claude-3-sonnet-latest",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_query}
            ]
        )
```

### 5.2 Visualization Suite

#### Timeline Analysis
```python
def _create_anomaly_visualization(self, data: Dict[str, Any], query: Dict) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Component Behavior',
            'Safety Status',
            'Anomaly Detection',
            'System Overview'
        )
    )
```

#### Professional Color Scheme
```python
COLOR_SCHEME = {
    'background': '#0e1117',
    'text': '#ffffff',
    'primary': '#2E86C1',    # Professional blue
    'secondary': '#85C1E9',  # Light blue
    'warning': '#F39C12',    # Warning orange
    'danger': '#E74C3C',     # Alert red
    'safe': '#2ECC71',       # Safe green
    'neutral': '#95A5A6'     # Neutral gray
}
```

#### Interactive Elements
- Component correlation heatmaps
- Safety status indicators
- Real-time trend analysis
- System overview dashboards

## 6. Technical Implementation

### 6.1 Core Components

#### ModelHandler
```python
class ModelHandler:
    def __init__(self, model_path: str = "best_model_pmt.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = EnhancedConfig()
        self.model_state = torch.load(model_path, map_location=self.device, weights_only=True)
```

Key functionalities:
- Model state management
- Data analysis
- Safety status checking
- Root cause detection
- Early warning generation

#### QueryProcessor
Handles natural language processing and query structuring:
```python
def process_query(self, query_text: str) -> Dict:
    # Convert natural language to structured query
    structured_query = self.query_processor.process_query(query_text)
    
    # Get relevant data window
    data_window = self._get_data_window(structured_query['timeframe'])
    
    # Analyze data
    analysis_results = self.model_handler.analyze_data(data_window, structured_query)
```

#### VisualizationManager
Creates professional-grade visualizations:
```python
class VisualizationManager:
    def create_visualization(self, data: Dict[str, Any], query: Dict) -> go.Figure:
        if 'detection' in query['query_type']:
            return self._create_anomaly_visualization(data, query)
        elif 'root_cause' in query['query_type']:
            return self._create_root_cause_visualization(data, query)
        elif 'early_warning' in query['query_type']:
            return self._create_early_warning_visualization(data, query)
```

### 6.2 Streamlit Interface

#### Setup and Configuration
```python
def main():
    st.set_page_config(
        page_title="Nuclear MCP Monitoring System",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
```

#### Custom Styling
```python
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: white;
    }
    .stTextInput > div > div > input {
        background-color: #262730;
        color: white;
        font-size: 16px;
        border: 1px solid #444;
    }
    </style>
""", unsafe_allow_html=True)
```
## 7. Usage Guide

### 7.1 Running in Google Colab

#### Training Pipeline Setup
1. Open the [Training Notebook](https://colab.research.google.com/drive/1Yi1865rOT80AGRy_FXlv9Flm185GD5Km)
2. Mount Google Drive and setup dependencies:
```python
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies
!pip install einops==0.4.0 matplotlib==3.7.0 patool==1.12 reformer-pytorch==1.4.4 
!pip install scikit-learn==1.2.2 sktime==0.16.1 sympy==1.12 tqdm==4.64.1
!pip install pyyaml wandb gluonts
```

3. Clone UniTS repository:
```python
!rm -rf UniTS
!git clone https://github.com/mims-harvard/UniTS.git
```

#### Analysis Tool Setup
1. Open the [Analysis Tool Notebook](https://colab.research.google.com/drive/1ZQeqUSfb4Bh3d5W1dTgBDDH9HiLwMF-J)
2. Install required packages:
```python
!pip install streamlit openai plotly pyngrok
```

3. Configure API keys:
```python
import os
os.environ['OPENROUTER_API_KEY'] = 'your_api_key_here'
```

### 7.2 Dataset Preparation

1. Create directory structure:
```python
!mkdir -p dataset/SMD
```

2. Copy dataset files:
```python
!cp '/content/drive/MyDrive/smd/train.txt' 'dataset/SMD/'
!cp '/content/drive/MyDrive/smd/test.txt' 'dataset/SMD/'
!cp '/content/drive/MyDrive/smd/test_label.txt' 'dataset/SMD/'
```

### 7.3 Example Queries

#### Anomaly Detection
```plaintext
"Analyze bearing temperature and pump motor load patterns during the last 30 minutes"
```
This query will:
- Monitor bearing temperature trends
- Check pump motor load correlations
- Compare against safety thresholds
- Generate anomaly alerts if needed

#### Root Cause Analysis
```plaintext
"What caused the spike in coolant pressure at 14:30?"
```
This query will:
- Analyze component correlations
- Identify primary contributors
- Show impact propagation
- Suggest potential causes

#### Early Warning Detection
```plaintext
"Check for early warning signs in flow rate and pressure systems"
```
This query will:
- Analyze trend patterns
- Calculate risk metrics
- Show predictive indicators
- Generate preventive recommendations

### 7.4 Interpreting Results

#### Visualization Panels
1. **Component Behavior**
   - Shows real-time parameter values
   - Displays safety thresholds
   - Highlights anomalous regions

2. **Safety Status**
   - Gauge showing overall system health
   - Color-coded status indicators
   - Compliance status

3. **Analysis Results**
   - Correlation heatmaps
   - Root cause indicators
   - Trend predictions

## 8. Example Outputs and Troubleshooting

### 8.1 Example Analysis Outputs

#### Anomaly Detection Output
```python
# Sample response structure
{
    'query': {
        'query_type': ['detection'],
        'primary_component': 'bearing_temperature',
        'timeframe': {'type': 'relative', 'value': 30, 'unit': 'minutes'}
    },
    'results': {
        'anomalies': [...],  # Binary indicators
        'reconstruction_error': [...],  # Error values
        'confidence_scores': [...],  # Detection confidence
        'safety_margins': {
            'bearing_temperature': 0.15,  # 15% margin from threshold
            'pump_motor_load': 0.22
        }
    }
}
```

#### Root Cause Analysis Output
```python
{
    'primary_components': [2, 5, 8],  # Index of problematic components
    'correlation_matrix': [...],  # Component correlations
    'impact_analysis': {
        'bearing_temperature': {
            'affected_components': ['pump_motor_load', 'coolant_pressure'],
            'correlation_strength': [0.85, 0.72]
        }
    }
}
```

### 8.2 Common Issues and Solutions - TBD

#### Model Loading Issues
```python
try:
    self.model_state = torch.load(model_path, map_location=self.device, weights_only=True)
except Exception as e:
    print(f"Error loading model: {str(e)}")
    # Possible solutions:
    # 1. Check model path
    # 2. Verify CUDA availability
    # 3. Ensure model compatibility
```

#### Data Processing Errors
Common issues and solutions:
1. **Missing Data**
   ```python
   # Handle missing values in preprocessing
   def handle_missing_data(self, data):
       if torch.isnan(data).any():
           # Interpolate missing values
           data = torch.nan_to_num(data, nan=self.default_value)
   ```

2. **Dimension Mismatch**
   ```python
   # Ensure correct dimensions
   assert data.shape[1] == self.config.input_dim, \
       f"Expected {self.config.input_dim} features, got {data.shape[1]}"
   ```

3. **Scale Issues**
   ```python
   # Check for scaling problems
   def check_scaling(self, data):
       if torch.abs(data).max() > 100:
           print("Warning: Unusual values detected, check scaling")
   ```

### 8.3 Performance Optimization

#### Memory Management
```python
def optimize_memory(self):
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Use batch processing for large datasets
    batch_size = min(self.config.batch_size, len(data))
```

#### Processing Speed
Tips for improving processing speed:
1. Use appropriate batch sizes
2. Implement data caching
3. Optimize visualization updates

Would you like me to continue with the Future Work and References sections?
