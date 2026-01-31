#include "operators/matmul.h"
#include "utils/operator_utils.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        auto shapeA = inputs[0]->getDims();
        auto shapeB = inputs[1]->getDims();
        int rankA = shapeA.size();
        int rankB = shapeB.size();

        if (rankA < 2 || rankB < 2) {
            return std::nullopt;
        }

        int rowA = shapeA[rankA - 2];
        int colA = shapeA[rankA - 1];
        int rowB = shapeB[rankB - 2];
        int colB = shapeB[rankB - 1];

        int actualM = transA ? colA : rowA;
        int actualK = transA ? rowA : colA;
        int K_in_B = transB ? colB : rowB;
        int actualN = transB ? rowB : colB;

        if (actualK != K_in_B) {
            return std::nullopt;
        }
        
        m = actualM;
        n = actualN;
        k = actualK;

        Shape batchA(shapeA.begin(), shapeA.end() - 2);
        Shape batchB(shapeB.begin(), shapeB.end() - 2);
        
        Shape resBatch = infer_broadcast(batchA, batchB);
        
        if (resBatch.empty() && (!batchA.empty() || !batchB.empty())) {
            if (std::max(batchA.size(), batchB.size()) > 0) return std::nullopt;
        }

        Shape outputShape = resBatch;
        outputShape.push_back(actualM);
        outputShape.push_back(actualN);

        return {{outputShape}};
    }

} // namespace infini