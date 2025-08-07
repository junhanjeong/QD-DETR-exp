#!/bin/bash

# AVIGATEFusionCustom Gate Analysis 빠른 시작 스크립트

echo "======================================================"
echo "AVIGATEFusionCustom Gate Analysis Quick Start"
echo "======================================================"

# 스크립트 실행 위치 확인
if [ ! -f "qd_detr/avigate_custom.py" ]; then
    echo "Error: Please run this script from the QD-DETR-exp root directory"
    exit 1
fi

# Python 환경 확인
python -c "import torch, matplotlib, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: Required packages not found. Please install:"
    echo "  pip install torch matplotlib numpy seaborn pandas"
    exit 1
fi

echo "Environment check passed!"
echo

# 메뉴 표시
echo "Choose an option:"
echo "1. Quick demo (recommended for first-time users)"
echo "2. Comprehensive test (full feature test)"
echo "3. Custom test (interactive)"
echo "4. Test specific gating type"
echo "5. Show help and documentation"

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo "Running quick demo..."
        python test_gate_analysis.py --quick
        ;;
    2)
        echo "Running comprehensive test..."
        echo "This may take a few minutes..."
        python test_gate_analysis.py --comprehensive
        ;;
    3)
        echo "Starting interactive test..."
        python test_gate_analysis.py
        ;;
    4)
        echo "Available gating types: global, clipwise, elementwise"
        read -p "Enter gating type: " gating_type
        
        # 임시 스크립트 생성
        cat > temp_test.py << EOF
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qd_detr.avigate_custom import AVIGATEFusionCustom
from gate_analysis_utils import GateAnalyzer
import torch

# 모델 생성
model = AVIGATEFusionCustom(
    vid_dim=256, aud_dim=256, hidden_dim=256,
    n_heads=8, num_layers=2, gating_type='$gating_type'
)

# 테스트 데이터
model.enable_gate_tracking()
for i in range(5):
    model.set_current_sample_id(f'test_{i}')
    video_feat = torch.randn(1, 75, 256)
    audio_feat = torch.randn(1, 75, 256)
    output = model(video_feat, audio_feat)

# 분석
analyzer = GateAnalyzer(model)
analyzer.analyze_all_samples('./test_${gating_type}_gating')
model.print_gate_summary()

print(f"Analysis for ${gating_type} gating completed!")
print("Results saved to: ./test_${gating_type}_gating/")
EOF
        
        python temp_test.py
        rm temp_test.py
        ;;
    5)
        echo "Opening documentation..."
        if command -v less &> /dev/null; then
            less GATE_ANALYSIS_README.md
        else
            cat GATE_ANALYSIS_README.md
        fi
        ;;
    *)
        echo "Invalid choice. Running quick demo..."
        python test_gate_analysis.py --quick
        ;;
esac

echo
echo "======================================================"
echo "Gate Analysis Complete!"
echo "======================================================"
echo
echo "Generated files and directories:"
ls -la | grep -E "(test_|gate_|quick_demo)"
echo
echo "Next steps:"
echo "1. Examine the generated analysis files"
echo "2. Check GATE_ANALYSIS_README.md for detailed usage"
echo "3. Integrate gate tracking into your actual model"
echo
echo "For more information:"
echo "  cat GATE_ANALYSIS_README.md"
echo "  python run_gate_analysis.py --help"
