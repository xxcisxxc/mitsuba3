#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/filesystem.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/string.h>
#include <mitsuba/render/film.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/imageblock.h>

#include <mutex>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _film-rxfilm:

Radio Frequency Receiver film (:monosp:`rxfilm`)
-------------------------------------------

.. pluginparameters::
 :extra-rows: 7

 * - width, height
   - |int|
   - Width and height of the camera sensor in pixels. Default: 768, 576)

 * - crop_offset_x, crop_offset_y, crop_width, crop_height
   - |int|
   - These parameters can optionally be provided to select a sub-rectangle
     of the output. In this case, only the requested regions
     will be rendered. (Default: Unused)

 * - sample_border
   - |bool|
   - If set to |true|, regions slightly outside of the film plane will also be sampled. This may
     improve the image quality at the edges, especially when using very large reconstruction
     filters. In general, this is not needed though. (Default: |false|, i.e. disabled)

 * - compensate
   - |bool|
   - If set to |true|, sample accumulation will be performed using Kahan-style
     error-compensated accumulation. This can be useful to avoid roundoff error
     when accumulating very many samples to compute reference solutions using
     single precision variants of Mitsuba. This feature is currently only supported
     in JIT variants and can make sample accumulation quite a bit more expensive.
     (Default: |false|, i.e. disabled)

 * - (Nested plugin)
   - :paramtype:`rfilter`
   - Reconstruction filter that should be used by the film. (Default: :monosp:`gaussian`, a windowed
     Gaussian filter)

 * - size
   - ``Vector2u``
   - Width and height of the camera sensor in pixels
   - |exposed|

 * - crop_size
   - ``Vector2u``
   - Size of the sub-rectangle of the output in pixels
   - |exposed|

 * - crop_offset
   - ``Point2u``
   - Offset of the sub-rectangle of the output in pixels
   - |exposed|

.. tabs::
    .. code-tab::  xml

        <film type="rxfilm">
            <integer name="width" value="128"/>
            <integer name="height" value="128"/>
        </film>

    .. code-tab:: python

        'type': 'rxfilm',
        'width': 128,
        'height': 128

 */

template <typename Float, typename Spectrum>
class RXFilm final : public Film<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Film, m_size, m_crop_size, m_crop_offset, m_sample_border,
                   m_filter, m_flags)
    MI_IMPORT_TYPES(ImageBlock)

    RXFilm(const Properties &props) : Base(props) {
        m_compensate = props.get<bool>("compensate", false);
        bool use_filter = props.get<bool>("use_filter", false);
        if (!use_filter)
            m_filter = nullptr;
    }

    size_t base_channels_count() const override {
        return 1;
    }

    size_t prepare(const std::vector<std::string> & /* aovs */) override {
        /* locked */
        std::lock_guard<std::mutex> lock(m_mutex);
        m_storage = new ImageBlock(m_crop_size, m_crop_offset, 1);

        return 1;
    }

    ref<ImageBlock> create_block(const ScalarVector2u &size, bool normalize,
                                 bool border) override {
        bool warn = false;

        bool default_config = dr::all(size == ScalarVector2u(0));

        return new ImageBlock(default_config ? m_crop_size : size,
                              default_config ? m_crop_offset : ScalarPoint2u(0),
                              (uint32_t)1, m_filter.get(),
                              border /* border */,
                              normalize /* normalize */,
                              false /* coalesce */,
                              m_compensate /* compensate */,
                              warn /* warn_negative */,
                              warn /* warn_invalid */);
    }

    void put_block(const ImageBlock *block) override {
        Assert(m_storage != nullptr);
        std::lock_guard<std::mutex> lock(m_mutex);
        m_storage->put_block(block);
    }

    void clear() override {
        if (m_storage)
            m_storage->clear();
    }

    TensorXf develop(bool /* raw */ = false) const override {
        if (!m_storage)
            Throw("No storage allocated, was prepare() called first?");

        std::lock_guard<std::mutex> lock(m_mutex);
        return m_storage->tensor();
    }

    ref<Bitmap> bitmap(bool /* raw */ = false) const override {
        if (!m_storage)
            Throw("No storage allocated, was prepare() called first?");

        std::lock_guard<std::mutex> lock(m_mutex);
        auto &&storage = dr::migrate(m_storage->tensor().array(), AllocType::Host);

        if constexpr (dr::is_jit_v<Float>)
            dr::sync_thread();

        Bitmap::PixelFormat source_fmt = Bitmap::PixelFormat::Y;

        ref<Bitmap> source = new Bitmap(
            source_fmt, struct_type_v<ScalarFloat>, m_storage->size(),
            m_storage->channel_count(), {}, (uint8_t *) storage.data());

        return source;
    }

    void write(const fs::path &path) const override {
        fs::path filename = path;

        #if !defined(_WIN32)
            Log(Info, "\U00002714  Developing \"%s\" ..", filename.string());
        #else
            Log(Info, "Developing \"%s\" ..", filename.string());
        #endif

        ref<Bitmap> source = bitmap();
        source->write(filename);
    }

    void schedule_storage() override {
        dr::schedule(m_storage->tensor());
    };

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "RXFilm[" << std::endl
            << "  size = " << m_size << "," << std::endl
            << "  crop_size = " << m_crop_size << "," << std::endl
            << "  crop_offset = " << m_crop_offset << "," << std::endl
            << "  sample_border = " << m_sample_border << "," << std::endl
            << "  compensate = " << m_compensate << "," << std::endl
            << "  filter = " << m_filter << "," << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
protected:
    bool m_compensate;
    ref<ImageBlock> m_storage;
    mutable std::mutex m_mutex;
};

MI_IMPLEMENT_CLASS_VARIANT(RXFilm, Film)
MI_EXPORT_PLUGIN(RXFilm, "RX Film")
NAMESPACE_END(mitsuba)
