#pragma once

#include <cstddef>
#include <iosfwd>
#include <string>

#include "nbody/body_system.hpp"
#include "nbody/diagnostics.hpp"
#include "nbody/types.hpp"

namespace nbody {

// Serie temporal de energia/momento (uma linha por amostra).
void write_metrics_header(std::ostream& out);
void write_metrics_row(std::ostream& out, std::size_t step, Real time, const Diagnostics& diag,
                       double elapsed_ms);

// Configuracao dos corpos (uma linha por corpo). Com append=true, acrescenta
// ao arquivo: encadeando snapshots a cada K passos voce monta uma animacao.
void write_snapshot_csv(const std::string& path, const BodySystem& system, std::size_t step,
                        Real time, bool append);

}  // namespace nbody
