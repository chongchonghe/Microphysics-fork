#ifndef actual_network_H
#define actual_network_H

#include <string_view>
#include <AMReX_REAL.H>
#include <AMReX_Array.H>

#include <fundamental_constants.H>
#include <network_properties.H>
#include <burn_type.H>

using namespace amrex;

void actual_network_init();

constexpr std::string_view network_name = "pynucastro-cxx";

namespace network
{
}

namespace Rates
{

    enum NetworkRates
    {
    };

}

AMREX_GPU_HOST_DEVICE AMREX_INLINE void balance_charge(burn_t &state) {

    // update the number density of electrons due to charge conservation
    state.xn[0] = -state.xn[3] - state.xn[7] + state.xn[1] + state.xn[12] +
                  state.xn[6] + state.xn[4] + state.xn[9] + 2.0 * state.xn[11];
};

#endif
