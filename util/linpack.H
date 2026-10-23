#ifndef LINPACK_H
#define LINPACK_H

#include <AMReX_REAL.H>
#include <AMReX_Array.H>

#include <ArrayUtilities.H>

#ifdef MICROPHYSICS_STATIC_PIVOT
// Partial pivoting with only compile-time indices: every loop is fully unrolled and the row swaps are done with
// "if (i == l)" selects instead of indexing by the runtime pivot row l. Same operations in the same order as the
// dynamic version below, so the results are identical, but a and b can stay in registers on GPUs.
template <int num_eqs>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void dgesl_static_pivot (const RArray2D& a, const IArray1D& pivot, RArray1D& b)
{
#pragma unroll
    for (int k = 1; k <= num_eqs - 1; ++k) {
        const int l = pivot(k);
        amrex::Real t = b(k);
#pragma unroll
        for (int i = k+1; i <= num_eqs; ++i) {
            if (i == l) {
                t = b(i);
                b(i) = b(k);
                b(k) = t;
            }
        }
#pragma unroll
        for (int j = k+1; j <= num_eqs; ++j) {
            b(j) += t * a(j,k);
        }
    }
#pragma unroll
    for (int kb = 1; kb <= num_eqs; ++kb) {
        const int k = num_eqs + 1 - kb;
        b(k) = b(k) / a(k,k);
        const amrex::Real t = -b(k);
#pragma unroll
        for (int j = 1; j <= k-1; ++j) {
            b(j) += t * a(j,k);
        }
    }
}

template <int num_eqs>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void dgefa_static_pivot (RArray2D& a, IArray1D& pivot, int& info)
{
    info = 0;
#pragma unroll
    for (int k = 1; k <= num_eqs - 1; ++k) {
        int l = k;
        amrex::Real dmax = std::abs(a(k,k));
#pragma unroll
        for (int i = k+1; i <= num_eqs; ++i) {
            const amrex::Real ai = std::abs(a(i,k));
            if (ai > dmax) {
                l = i;
                dmax = ai;
            }
        }
        pivot(k) = static_cast<short>(l);

        amrex::Real alk = a(k,k); // a(l,k)
#pragma unroll
        for (int i = k+1; i <= num_eqs; ++i) {
            if (i == l) {
                alk = a(i,k);
            }
        }

        if (alk != 0.0e0_rt) {
            // interchange a(l,k) and a(k,k) if necessary
#pragma unroll
            for (int i = k+1; i <= num_eqs; ++i) {
                if (i == l) {
                    a(i,k) = a(k,k);
                    a(k,k) = alk;
                }
            }

            // compute multipliers
            const amrex::Real tm = -1.0e0_rt / a(k,k);
#pragma unroll
            for (int j = k+1; j <= num_eqs; ++j) {
                a(j,k) *= tm;
            }

            // row elimination with column indexing
#pragma unroll
            for (int j = k+1; j <= num_eqs; ++j) {
                amrex::Real t = a(k,j); // a(l,j)
#pragma unroll
                for (int i = k+1; i <= num_eqs; ++i) {
                    if (i == l) {
                        t = a(i,j);
                        a(i,j) = a(k,j);
                        a(k,j) = t;
                    }
                }
#pragma unroll
                for (int i = k+1; i <= num_eqs; ++i) {
                    a(i,j) += t * a(i,k);
                }
            }
        } else {
            info = k;
        }
    }

    pivot(num_eqs) = static_cast<short>(num_eqs);

    if (a(num_eqs,num_eqs) == 0.0e0_rt) {
        info = num_eqs;
    }
}
#endif

template <int num_eqs, bool allow_pivot>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void dgesl (const RArray2D& a, const IArray1D& pivot, RArray1D& b)
{
#ifdef MICROPHYSICS_STATIC_PIVOT
    if constexpr (allow_pivot) {
        dgesl_static_pivot<num_eqs>(a, pivot, b);
        return;
    }
#endif

    constexpr int nm1 = num_eqs - 1;

    // solve a * x = b
    // first solve l * y = b
    if constexpr (nm1 >= 1) {
        for (int k = 1; k <= nm1; ++k) {

            amrex::Real t{};
            if constexpr (allow_pivot) {
                int l = pivot(k);
                t = b(l);
                if (l != k) {
                    b(l) = b(k);
                    b(k) = t;
                }
            } else {
                t = b(k);
            }

            AMREX_UNROLL_LOOP(NET_LOOP_UNROLL_LEN)
            for (int j = k+1; j <= num_eqs; ++j) {
                b(j) += t * a(j,k);
            }
        }
    }

    // now solve u * x = y
    for (int kb = 1; kb <= num_eqs; ++kb) {

        const int k = num_eqs + 1 - kb;
        b(k) = b(k) / a(k,k);
        amrex::Real t = -b(k);
        AMREX_UNROLL_LOOP(NET_LOOP_UNROLL_LEN)
        for (int j = 1; j <= k-1; ++j) {
            b(j) += t * a(j,k);
        }
    }

}



template <int num_eqs, bool allow_pivot>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void dgefa (RArray2D& a, IArray1D& pivot, int& info)
{

    // dgefa factors a matrix by gaussian elimination.
    // a is returned in the form a = l * u where
    // l is a product of permutation and unit lower
    // triangular matrices and u is upper triangular.

    // gaussian elimination with partial pivoting

#ifdef MICROPHYSICS_STATIC_PIVOT
    if constexpr (allow_pivot) {
        dgefa_static_pivot<num_eqs>(a, pivot, info);
        return;
    }
#endif

    info = 0;
    constexpr int nm1 = num_eqs - 1;

    amrex::Real t;

    if constexpr (nm1 >= 1) {

        for (int k = 1; k <= nm1; ++k) {

            // find l = pivot index
            int l = k;

            if constexpr (allow_pivot) {
                amrex::Real dmax = std::abs(a(k,k));
                AMREX_UNROLL_LOOP(NET_LOOP_UNROLL_LEN)
                for (int i = k+1; i <= num_eqs; ++i) {
                    amrex::Real ai = std::abs(a(i, k));
                    if (ai > dmax) {
                        l = i;
                        dmax = ai;
                    }
                }

                pivot(k) = static_cast<short>(l);
            }

            // zero pivot implies this column already triangularized
            if (a(l,k) != 0.0e0_rt) {

                if constexpr (allow_pivot) {
                    // interchange if necessary
                    if (l != k) {
                        t = a(l,k);
                        a(l,k) = a(k,k);
                        a(k,k) = t;
                    }
                }

                // compute multipliers
                t = -1.0e0_rt / a(k,k);
                AMREX_UNROLL_LOOP(NET_LOOP_UNROLL_LEN)
                for (int j = k+1; j <= num_eqs; ++j) {
                    a(j,k) *= t;
                }

                // row elimination with column indexing
                for (int j = k+1; j <= num_eqs; ++j) {
                    t = a(l,j);

                    if constexpr (allow_pivot) {
                        if (l != k) {
                            a(l,j) = a(k,j);
                            a(k,j) = t;
                        }
                    }

                    AMREX_UNROLL_LOOP(NET_LOOP_UNROLL_LEN)
                    for (int i = k+1; i <= num_eqs; ++i) {
                        a(i,j) += t * a(i,k);
                    }
                }

            } else {
                info = k;
            }

        }

    }

    if constexpr (allow_pivot) {
        pivot(num_eqs) = static_cast<short>(num_eqs);
    }

    if (a(num_eqs,num_eqs) == 0.0e0_rt) {
        info = num_eqs;
    }

}

#endif
