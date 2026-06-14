// =============================================================================
// TurnipPrices_official.cpp
// -----------------------------------------------------------------------------
// ACNH (Animal Crossing: New Horizons) 大頭菜價格 官方演算法 — 參考實作。
//
// 本檔為「社群 datamine 出的官方演算法」之乾淨重建, 用來與 turnip_sim.py 逐行對照。
// 原始反組譯成果歸功於 Ninji (Treeki) 與 simontime/Resead (sead::Random)。
// 原始 Gist:  https://gist.github.com/Treeki/85be14d297c80c8b3c0a76375743325b
// 詳見 sources.md。
//
// 重點: 遊戲以 *單精度 float* 計算, 且 sead::Random 為 32-bit。移植時務必比照。
// =============================================================================

#include <cstdint>

namespace sead {
class Random {
private:
    uint32_t mContext[4];
public:
    void init(uint32_t seed) {
        mContext[0] = 0x6C078965 * (seed        ^ (seed        >> 30)) + 1;
        mContext[1] = 0x6C078965 * (mContext[0] ^ (mContext[0] >> 30)) + 2;
        mContext[2] = 0x6C078965 * (mContext[1] ^ (mContext[1] >> 30)) + 3;
        mContext[3] = 0x6C078965 * (mContext[2] ^ (mContext[2] >> 30)) + 4;
    }

    uint32_t getU32() {
        uint32_t n = mContext[0] ^ (mContext[0] << 11);
        mContext[0] = mContext[1];
        mContext[1] = mContext[2];
        mContext[2] = mContext[3];
        mContext[3] = n ^ (n >> 8) ^ mContext[3] ^ (mContext[3] >> 19);
        return mContext[3];
    }
};
} // namespace sead

struct TurnipPrices {
    int32_t      basePrice;
    int32_t      sellPrices[14];   // [0],[1] = 週日(買), [2..13] = 週一~週六共12個賣價
    uint32_t     whatPattern;      // 上週 pattern, 產生後更新為本週
    sead::Random rng;

    int randint(int min, int max) {
        return (int)(((uint64_t)rng.getU32() * (uint64_t)(max - min + 1)) >> 32) + min;
    }

    float randfloat(float a, float b) {
        uint32_t val = 0x3F800000 | (rng.getU32() >> 9);
        float fval = *(float *)(&val);          // 落在 [1.0, 2.0)
        return a + ((fval - 1.0f) * (b - a));   // 注意: a 可大於 b
    }

    bool randbool() {
        return rng.getU32() & 0x80000000;
    }

    int intceil(float val) {
        return (int)(val + 0.99999f);
    }

    void calculate() {
        basePrice = randint(90, 110);
        int chance = randint(0, 99);

        // ---- pattern 轉移表 ----
        int nextPattern;
        if (whatPattern >= 4) {
            nextPattern = 2;
        } else {
            switch (whatPattern) {
            case 0:
                nextPattern = (chance < 20) ? 0 : (chance < 50) ? 1 : (chance < 65) ? 2 : 3;
                break;
            case 1:
                nextPattern = (chance < 50) ? 0 : (chance < 55) ? 1 : (chance < 75) ? 2 : 3;
                break;
            case 2:
                nextPattern = (chance < 25) ? 0 : (chance < 70) ? 1 : (chance < 75) ? 2 : 3;
                break;
            default: // case 3
                nextPattern = (chance < 45) ? 0 : (chance < 70) ? 1 : (chance < 85) ? 2 : 3;
                break;
            }
        }
        whatPattern = nextPattern;

        int work;
        float rate;

        switch (whatPattern) {
        case 0: {
            // Fluctuating: 高 / 降 / 高 / 降 / 高
            work = 2;
            int decPhaseLen1 = randbool() ? 3 : 2;
            int decPhaseLen2 = 5 - decPhaseLen1;
            int hiPhaseLen1 = randint(0, 6);
            int hiPhaseLen2and3 = 7 - hiPhaseLen1;
            int hiPhaseLen3 = randint(0, hiPhaseLen2and3 - 1);

            for (int i = 0; i < hiPhaseLen1; i++)
                sellPrices[work++] = intceil(randfloat(0.9, 1.4) * basePrice);

            rate = randfloat(0.8, 0.6);
            for (int i = 0; i < decPhaseLen1; i++) {
                sellPrices[work++] = intceil(rate * basePrice);
                rate -= 0.04;
                rate -= randfloat(0, 0.06);
            }

            for (int i = 0; i < (hiPhaseLen2and3 - hiPhaseLen3); i++)
                sellPrices[work++] = intceil(randfloat(0.9, 1.4) * basePrice);

            rate = randfloat(0.8, 0.6);
            for (int i = 0; i < decPhaseLen2; i++) {
                sellPrices[work++] = intceil(rate * basePrice);
                rate -= 0.04;
                rate -= randfloat(0, 0.06);
            }

            for (int i = 0; i < hiPhaseLen3; i++)
                sellPrices[work++] = intceil(randfloat(0.9, 1.4) * basePrice);
            break;
        }

        case 1: {
            // Large spike: 先降 -> 爆衝 -> 隨機低
            int peakStart = randint(3, 9);
            rate = randfloat(0.9, 0.85);
            for (work = 2; work < peakStart; work++) {
                sellPrices[work] = intceil(rate * basePrice);
                rate -= 0.03;
                rate -= randfloat(0, 0.02);
            }
            sellPrices[work++] = intceil(randfloat(0.9, 1.4) * basePrice);
            sellPrices[work++] = intceil(randfloat(1.4, 2.0) * basePrice);
            sellPrices[work++] = intceil(randfloat(2.0, 6.0) * basePrice);   // 最高點
            sellPrices[work++] = intceil(randfloat(1.4, 2.0) * basePrice);
            sellPrices[work++] = intceil(randfloat(0.9, 1.4) * basePrice);
            for (; work < 14; work++)
                sellPrices[work] = intceil(randfloat(0.4, 0.9) * basePrice);
            break;
        }

        case 2: {
            // Decreasing: 整週遞減
            rate = 0.9;
            rate -= randfloat(0, 0.05);
            for (work = 2; work < 14; work++) {
                sellPrices[work] = intceil(rate * basePrice);
                rate -= 0.03;
                rate -= randfloat(0, 0.02);
            }
            break;
        }

        case 3: {
            // Small spike: 降 -> 小尖峰 -> 降
            int peakStart = randint(2, 9);
            rate = randfloat(0.9, 0.4);
            for (work = 2; work < peakStart; work++) {
                sellPrices[work] = intceil(rate * basePrice);
                rate -= 0.03;
                rate -= randfloat(0, 0.02);
            }
            sellPrices[work++] = intceil(randfloat(0.9, 1.4) * basePrice);
            sellPrices[work++] = intceil(randfloat(0.9, 1.4) * basePrice);
            rate = randfloat(1.4, 2.0);
            sellPrices[work++] = intceil(randfloat(1.4, rate) * basePrice) - 1;
            sellPrices[work++] = intceil(rate * basePrice);                   // 最高點
            sellPrices[work++] = intceil(randfloat(1.4, rate) * basePrice) - 1;
            if (work < 14) {
                rate = randfloat(0.9, 0.4);
                for (; work < 14; work++) {
                    sellPrices[work] = intceil(rate * basePrice);
                    rate -= 0.03;
                    rate -= randfloat(0, 0.02);
                }
            }
            break;
        }
        }

        sellPrices[0] = basePrice;  // 週日: 以買價表示
        sellPrices[1] = basePrice;
    }
};
