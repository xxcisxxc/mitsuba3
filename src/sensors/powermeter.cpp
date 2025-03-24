#include "drjit/sphere.h"
#include "mitsuba/core/fwd.h"
#include "mitsuba/core/warp.h"
#include <mitsuba/core/bbox.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/render/sensor.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _sensor-powermeter:

PowerMeter camera (:monosp:`powermeter`)
--------------------------------------------------

.. pluginparameters::

 * - to_world
   - |transform|
   - Specifies an optional camera-to-world transformation.
     (Default: none (i.e. camera space = world space))
   - |exposed|

 * - near_clip, far_clip
   - |float|
   - Distance to the near/far clip planes. (Default: :monosp:`near_clip=1e-2`
(i.e. :monosp:`0.01`) and :monosp:`far_clip=1e4` (i.e. :monosp:`10000`))
   - |exposed|

 * - srf
   - |spectrum|
   - Sensor Response Function that defines the :ref:`spectral sensitivity
<explanation_srf_sensor>` of the sensor (Default: :monosp:`none`)

 * - fov
   - |float|
   - Field of view of each pixel in degrees. Must be in the range [0, 180].
     (Default: :monosp:`180`)
   - |exposed|

This plugin implements a simple orthographic power meter, i.e. a sensor
based on an orthographic projection without any form of perspective.
It can be thought of as a bunch of sensors that forms a hypothetical plane that
measures the radiance within a cone of angles around the Z axis. By default,
this is the region $[-1, 1]^2$ inside the XY-plane facing along the positive Z
direction. Transformed versions can be instantiated e.g. as follows:

The exact camera position and orientation is most easily expressed using the
:monosp:`lookat` tag, i.e.:

.. tabs::
    .. code-tab:: xml

        <sensor type="powermeter">
            <transform name="to_world">
                <!-- Resize the sensor plane to 20x20 world space units -->
                <scale x="10" y="10"/>

                <!-- Move and rotate the camera so that looks from (1, 1, 1) to
(1, 2, 1) and the direction (0, 0, 1) points "up" in the output image -->
                <lookat origin="1, 1, 1" target="1, 2, 1" up="0, 0, 1"/>
            </transform>
        </sensor>

    .. code-tab:: python

        'type': 'powermeter',
        'to_world': mi.ScalarTransform4f().look_at(
            origin=[1, 1, 1],
            target=[1, 2, 1],
            up=[0, 0, 1]
        ) @ mi.ScalarTransform4f().scale([10, 10, 1])

 */

template <typename Float, typename Spectrum>
class PowerMeter final : public ProjectiveCamera<Float, Spectrum> {
public:
    MI_IMPORT_BASE(ProjectiveCamera, m_to_world, m_needs_sample_3, m_film,
                   m_sampler, m_resolution, m_shutter_open, m_shutter_open_time,
                   m_near_clip, m_far_clip, sample_wavelengths)
    MI_IMPORT_TYPES()

    PowerMeter(const Properties &props) : Base(props) {
        update_camera_transforms();
        m_needs_sample_3 = true;

        // double fov = props.get<double>("fov", 180.0);

        // if (fov < 0.0 || fov > 180.0)
        //     Throw(
        //         "The horizontal field of view must be in the range [0,
        //         180]!");
        // m_fov     = (ScalarFloat) fov;
        // m_cos_fov = dr::cos(dr::deg_to_rad(m_fov / 2.f));
        // dr::make_opaque(m_fov, m_cos_fov);
    }

    void traverse(TraversalCallback *callback) override {
        Base::traverse(callback);
        callback->put_parameter("to_world", *m_to_world.ptr(),
                                +ParamFlags::NonDifferentiable);
        // callback->put_parameter("fov", m_fov,
        // +ParamFlags::NonDifferentiable);
    }

    void parameters_changed(const std::vector<std::string> &keys) override {
        Base::parameters_changed(keys);
        update_camera_transforms();

        // m_cos_fov = dr::cos(dr::deg_to_rad(m_fov / 2.f));
        // dr::make_opaque(m_fov, m_cos_fov);
    }

    void update_camera_transforms() {
        m_camera_to_sample = orthographic_projection(
            m_film->size(), m_film->crop_size(), m_film->crop_offset(),
            Float(m_near_clip), Float(m_far_clip));

        m_sample_to_camera = m_camera_to_sample.inverse();

        // Position differentials on the near plane
        m_dx = m_sample_to_camera * Point3f(1.f / m_resolution.x(), 0.f, 0.f) -
               m_sample_to_camera * Point3f(0.f);
        m_dy = m_sample_to_camera * Point3f(0.f, 1.f / m_resolution.y(), 0.f) -
               m_sample_to_camera * Point3f(0.f);

        /* Precompute some data for importance(). Please
           look at that function for further details. */
        Point3f pmin(m_sample_to_camera * Point3f(0.f, 0.f, 0.f)),
            pmax(m_sample_to_camera * Point3f(1.f, 1.f, 0.f));

        m_image_rect.reset();
        m_image_rect.expand(Point2f(pmin.x(), pmin.y()) / pmin.z());
        m_image_rect.expand(Point2f(pmax.x(), pmax.y()) / pmax.z());
        m_normalization = 1.f / m_image_rect.volume();

        dr::make_opaque(m_camera_to_sample, m_sample_to_camera, m_dx, m_dy,
                        m_normalization);
        // dr::make_opaque(m_camera_to_sample, m_sample_to_camera, m_dx, m_dy);
    }

    std::pair<Ray3f, Spectrum> sample_ray(Float time, Float wavelength_sample,
                                          const Point2f &position_sample,
                                          const Point2f &aperture_sample,
                                          Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);

        auto [wavelengths, wav_weight] = sample_wavelengths(
            dr::zeros<SurfaceInteraction3f>(), wavelength_sample, active);
        Ray3f ray;
        ray.time        = time;
        ray.wavelengths = wavelengths;

        // Compute the sample position on the near plane (local camera space).
        Point3f near_p = m_sample_to_camera *
                         Point3f(position_sample.x(), position_sample.y(), 0.f);

        // Vector3f sampled_d =
        //     warp::square_to_uniform_cone(aperture_sample, m_cos_fov);
        Vector3f sampled_d =
            warp::square_to_uniform_hemisphere(aperture_sample);

        ray.o    = m_to_world.value() * near_p;
        ray.d    = dr::normalize(m_to_world.value() * sampled_d);
        ray.maxt = m_far_clip - m_near_clip;

        return { ray, wav_weight * dr::Pi<ScalarFloat> };
    }

    std::pair<RayDifferential3f, Spectrum> sample_ray_differential(
        Float time, Float wavelength_sample, const Point2f &position_sample,
        const Point2f &aperture_sample, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);

        auto [wavelengths, wav_weight] = sample_wavelengths(
            dr::zeros<SurfaceInteraction3f>(), wavelength_sample, active);
        RayDifferential3f ray;
        ray.time        = time;
        ray.wavelengths = wavelengths;

        // Compute the sample position on the near plane (local camera space).
        Point3f near_p = m_sample_to_camera *
                         Point3f(position_sample.x(), position_sample.y(), 0.f);

        // Vector3f sampled_d =
        //     warp::square_to_uniform_cone(aperture_sample, m_cos_fov);
        Vector3f sampled_d =
            warp::square_to_uniform_hemisphere(aperture_sample);

        ray.o    = m_to_world.value() * near_p;
        ray.d    = dr::normalize(m_to_world.value() * sampled_d);
        ray.maxt = m_far_clip - m_near_clip;

        ray.o_x = m_to_world.value() * (near_p + m_dx);
        ray.o_y = m_to_world.value() * (near_p + m_dy);
        ray.d_x = ray.d_y     = ray.d;
        ray.has_differentials = true;

        return { ray, wav_weight * dr::Pi<ScalarFloat> };
    }

    std::pair<DirectionSample3f, Spectrum>
    sample_direction(const Interaction3f &it, const Point2f &sample,
                     Mask active) const override {
        // Transform the reference point into the local coordinate system
        Transform4f trafo = m_to_world.value();
        Point3f ref_p     = trafo.inverse().transform_affine(it.p);

        // Check if it is outside of the clip range
        DirectionSample3f ds = dr::zeros<DirectionSample3f>();
        ds.pdf               = 0.f;
        active &= (ref_p.z() >= m_near_clip) && (ref_p.z() <= m_far_clip);
        if (dr::none_or<false>(active))
            return { ds, dr::zeros<Spectrum>() };

        // Compute the viewable area on the near plane
        // Float tan_fov = dr::tan(dr::deg_to_rad(m_fov / 2.f));
        // Float offset  = dr::clip(2 * ref_p.z() * tan_fov, 0.f, 2.f);

        // Vector2f p_min = Vector2f(ref_p.x() - offset, ref_p.y() - offset);
        // Vector2f p_max = Vector2f(ref_p.x() + offset, ref_p.y() + offset);
        // active &= !(p_min.x() >= 1.f || p_min.y() >= 1.f || p_max.x() <= -1.f
        // ||
        //             p_max.y() <= -1.f);
        // if (dr::none_or<false>(active))
        //     return { ds, dr::zeros<Spectrum>() };

        // p_min           = dr::clip(p_min, -1.f, 1.f);
        // p_max           = dr::clip(p_max, -1.f, 1.f);
        // Vector2f clip   = p_max - p_min;
        // Vector2f center = (p_min + p_max) / 2.f;
        // Transform4f trafo_sample =
        //     Transform4f::translate(Vector3f(center.x(), center.y(), 0.f)) *
        //     Transform4f::scale(Vector3f(clip.x(), clip.y(), 1.f));

        // Point3f sample_p = trafo_sample * m_sample_to_camera *
        //                    Point3f(sample.x(), sample.y(), 0.f);
        Point3f sample_p =
            m_sample_to_camera * Point3f(sample.x(), sample.y(), 0.f);
        // Point3f screen_sample = m_camera_to_sample * sample_p;
        // ds.uv                 = dr::head<2>(screen_sample) * m_resolution;
        ds.uv = sample * m_resolution;

        Vector3f local_d = ref_p - sample_p;
        Float dist       = dr::norm(local_d);
        Float inv_dist   = dr::rcp(dist);
        local_d *= inv_dist;

        // Compute importance value
        Float ct = Frame3f::cos_theta(local_d), inv_ct = dr::rcp(ct);
        // Float importance =
        //     dr::select(active, m_normalization * inv_ct * inv_ct * inv_ct,
        //     0.f);
        Float importance = dr::select(active, m_normalization * inv_ct, 0.f);

        ds.p    = trafo.transform_affine(sample_p);
        ds.d    = (ds.p - it.p) * inv_dist;
        ds.dist = dist;
        ds.n    = trafo * Vector3f(0.0f, 0.0f, 1.0f);
        // ds.pdf  = dr::select(
        //     active, warp::square_to_uniform_cone_pdf(local_d, m_cos_fov),
        //     Float(0.f));
        ds.pdf = importance * dist * dist;

        Spectrum weight = dr::select(active, dr::rcp(ds.pdf), Spectrum(0.f));

        return { ds, weight };
    }

    ScalarBoundingBox3f bbox() const override {
        ScalarPoint3f p = m_to_world.scalar() * ScalarPoint3f(0.f);
        return ScalarBoundingBox3f(p, p);
    }

    std::string to_string() const override {
        using string::indent;

        std::ostringstream oss;
        oss << "PowerMeter[" << std::endl
            << "  fov = " << m_fov << "," << std::endl
            << "  near_clip = " << m_near_clip << "," << std::endl
            << "  far_clip = " << m_far_clip << "," << std::endl
            << "  film = " << indent(m_film) << "," << std::endl
            << "  sampler = " << indent(m_sampler) << "," << std::endl
            << "  resolution = " << m_resolution << "," << std::endl
            << "  shutter_open = " << m_shutter_open << "," << std::endl
            << "  shutter_open_time = " << m_shutter_open_time << ","
            << std::endl
            << "  world_transform = " << indent(m_to_world) << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
private:
    Transform4f m_camera_to_sample;
    Transform4f m_sample_to_camera;
    BoundingBox2f m_image_rect;
    Float m_normalization;
    Vector3f m_dx, m_dy;
    Float m_fov;
    Float m_cos_fov;
};

MI_IMPLEMENT_CLASS_VARIANT(PowerMeter, ProjectiveCamera)
MI_EXPORT_PLUGIN(PowerMeter, "PowerMeter");
NAMESPACE_END(mitsuba)
