document.addEventListener("DOMContentLoaded", function () {

    const pupil = document.querySelector(".pupil");
    const eye = document.querySelector(".eye-animation");

    if (!pupil || !eye) return;

    document.addEventListener("mousemove", function (e) {

        const rect = eye.getBoundingClientRect();
        const eyeX = rect.left + rect.width / 2;
        const eyeY = rect.top + rect.height / 2;

        const angle = Math.atan2(e.clientY - eyeY, e.clientX - eyeX);

        const radius = 30;
        const x = radius * Math.cos(angle);
        const y = radius * Math.sin(angle);

        pupil.style.transform = `translate(${x}px, ${y}px)`;

    });

});